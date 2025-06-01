import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import flwr as fl
import numpy as np
from opacus import PrivacyEngine
import json

# Define the CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load and partition MNIST dataset
def load_data(num_clients=10):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    # Simulate non-IID data
    indices = np.arange(len(trainset))
    labels = np.array(trainset.targets)
    client_indices = [[] for _ in range(num_clients)]
    for i in range(10):
        class_indices = indices[labels == i]
        np.random.shuffle(class_indices)
        for j, idx in enumerate(class_indices):
            client_idx = (i + j % 3) % num_clients
            client_indices[client_idx].append(idx)
    
    client_loaders = [DataLoader(Subset(trainset, indices), batch_size=32, shuffle=True) for indices in client_indices]
    test_loader = DataLoader(testset, batch_size=32, shuffle=False)
    return client_loaders, test_loader

# Flower client
class MNISTClient(fl.client.NumPyClient):
    def __init__(self, cid, model, train_loader, test_loader):
        self.cid = cid
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.privacy_engine = PrivacyEngine()
        self.model, self.optimizer, self.train_loader = self.privacy_engine.make_private(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.train_loader,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
        )
    
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        for epoch in range(1):
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                self.optimizer.step()
        return self.get_parameters(config={}), len(self.train_loader.dataset), {}
    
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        loss, correct = 0, 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss += F.cross_entropy(output, target, reduction='sum').item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target.view_as(pred)).sum().item()
        loss /= len(self.test_loader.dataset)
        accuracy = correct / len(self.test_loader.dataset)
        return loss, len(self.test_loader.dataset), {"accuracy": accuracy}

# Custom strategy to collect metrics
class CustomFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics = {"rounds": [], "accuracy": [], "loss": []}
    
    def aggregate_evaluate(self, server_round, results, failures):
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)
        accuracy = np.mean([r.metrics["accuracy"] for _, r in results])
        self.metrics["rounds"].append(server_round)
        self.metrics["accuracy"].append(accuracy)
        self.metrics["loss"].append(loss)
        print(f"Round {server_round}: accuracy={accuracy:.4f}, loss={loss:.4f}")
        return loss, metrics
    
    def save_metrics(self, filename="metrics.json"):
        with open(filename, 'w') as f:
            json.dump(self.metrics, f)

# Start Flower server
def main():
    client_loaders, test_loader = load_data()
    strategy = CustomFedAvg(
        fraction_fit=0.5,
        fraction_evaluate=0.5,
        min_fit_clients=5,
        min_evaluate_clients=5,
        min_available_clients=10,
    )
    fl.server.start_server(
        server_address="0.0.0.0:8084",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )
    strategy.save_metrics()

def client_fn(cid):
    model = SimpleCNN()
    client_loaders, test_loader = load_data()
    return MNISTClient(cid, model, client_loaders[int(cid)], test_loader)

if __name__ == "__main__":
    main()