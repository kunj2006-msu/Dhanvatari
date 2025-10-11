import requests
import json

# Test the interactive conversation system
url = "http://localhost:8000/chat"

# Test 1: Initial fever + cough message
print("=== Test 1: Initial Message ===")
data1 = {
    "messages": [
        {"role": "user", "content": "I have fever from last night and also cough"}
    ]
}

response1 = requests.post(url, json=data1)
print(f"Response 1: {response1.json()['response'][:200]}...")

# Test 2: Follow-up response
print("\n=== Test 2: Follow-up Message ===")
data2 = {
    "messages": [
        {"role": "user", "content": "I have fever from last night and also cough"},
        {"role": "assistant", "content": "Previous response..."},
        {"role": "user", "content": "My temperature is 102F and I have dry cough"}
    ]
}

response2 = requests.post(url, json=data2)
print(f"Response 2: {response2.json()['response'][:200]}...")

# Test 3: Third interaction
print("\n=== Test 3: Third Message ===")
data3 = {
    "messages": [
        {"role": "user", "content": "I have fever from last night and also cough"},
        {"role": "assistant", "content": "Previous response..."},
        {"role": "user", "content": "My temperature is 102F and I have dry cough"},
        {"role": "assistant", "content": "Previous response..."},
        {"role": "user", "content": "I tried the turmeric milk and it helped a bit"}
    ]
}

response3 = requests.post(url, json=data3)
print(f"Response 3: {response3.json()['response'][:200]}...")
