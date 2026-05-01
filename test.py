import requests

data = {
    "sessions": 12400,
    "conversions": 310,
    "ad_spend": 2500,
    "revenue": 18700,
    "period": "Апрель 2025"
}

response = requests.post("http://localhost:8000/generate-report", json=data)
print(response.json())