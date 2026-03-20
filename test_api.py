import requests

url = "http://127.0.0.1:5000/predict"

data = {
    "features": [90, 40, 40, 20, 80, 6.5, 200]
}

response = requests.post(url, json=data)

print(response.json())