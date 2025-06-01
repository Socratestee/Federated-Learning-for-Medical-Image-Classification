import flwr as fl
import sys
from main import client_fn

if __name__ == "__main__":
    cid = sys.argv[1]  # Client ID passed as argument
    fl.client.start_numpy_client(server_address="localhost:8084", client=client_fn(cid))