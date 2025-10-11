import requests
import json

# Test the chat endpoint
url = "http://localhost:8000/chat"
data = {
    "messages": [
        {"role": "user", "content": "I have a headache"}
    ]
}

try:
    response = requests.post(url, json=data)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
except Exception as e:
    print(f"Error: {e}")
