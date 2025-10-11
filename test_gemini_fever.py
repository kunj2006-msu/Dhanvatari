import requests
import json

# Test Gemini with fever and cough
url = "http://localhost:8000/chat"

print("=== Testing Gemini AI with Fever + Cough ===")
data = {
    "messages": [
        {"role": "user", "content": "I have fever from last night and also cough"}
    ]
}

response = requests.post(url, json=data)
print("Gemini Response:")
print(response.json()['response'])
print("\n" + "="*50)

# Test follow-up
print("\n=== Follow-up Response ===")
data2 = {
    "messages": [
        {"role": "user", "content": "I have fever from last night and also cough"},
        {"role": "assistant", "content": "Previous response..."},
        {"role": "user", "content": "My temperature is 102F and I have dry cough"}
    ]
}

response2 = requests.post(url, json=data2)
print("Gemini Follow-up:")
print(response2.json()['response'])
