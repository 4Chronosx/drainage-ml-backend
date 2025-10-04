import requests # type: ignore

url = "http://localhost:8000/predict-25yr"
payload = {"point": [1.5, 0.5, 300.0, 0.000014]}

response = requests.post(url, json=payload)

print("Status:", response.status_code)
print("Response:", response.json())
