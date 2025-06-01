import json

# Load metrics
with open('metrics.json', 'r') as f:
    metrics = json.load(f)

# Generate Chart.js configuration
chart_config = {
    "type": "line",
    "data": {
        "labels": metrics["rounds"],
        "datasets": [
            {
                "label": "Accuracy",
                "data": metrics["accuracy"],
                "borderColor": "rgb(75, 192, 192)",
                "backgroundColor": "rgba(75, 192, 192, 0.2)",
                "fill": False,
                "tension": 0.4
            },
            {
                "label": "Loss",
                "data": metrics["loss"],
                "borderColor": "rgb(255, 99, 132)",
                "backgroundColor": "rgba(255, 99, 132, 0.2)",
                "fill": False,
                "tension": 0.4
            }
        ]
    },
    "options": {
        "responsive": True,
        "plugins": {
            "title": {
                "display": True,
                "text": "Federated Learning: Accuracy and Loss Over Rounds"
            }
        },
        "scales": {
            "x": {
                "title": {
                    "display": True,
                    "text": "Round"
                }
            },
            "y": {
                "title": {
                    "display": True,
                    "text": "Value"
                },
                "beginAtZero": True
            }
        }
    }
}

# Print Chart.js configuration
print("```chartjs")
print(json.dumps(chart_config, indent=2))
print("```")