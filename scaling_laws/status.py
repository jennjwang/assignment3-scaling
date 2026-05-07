import requests
import json
import os   
import json

def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

API_BASE_URL = "http://hyperturing.stanford.edu:8000"
API_KEY = "50030271"
headers = {"X-API-Key": API_KEY}
budget = requests.get(f"{API_BASE_URL}/budget", headers=headers).json()
print(budget)

experiments = requests.get(f"{API_BASE_URL}/experiments", headers=headers).json()
# print(experiments)

save_json(experiments, "scaling_laws/experiments.json")