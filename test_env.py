import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check if API key is loaded
api_key = os.getenv("GEMINI_API_KEY")
print(f"API Key found: {api_key is not None}")
if api_key:
    print(f"API Key starts with: {api_key[:10]}...")
else:
    print("No API key found")

# List all environment variables that start with GEMINI
for key, value in os.environ.items():
    if "GEMINI" in key:
        print(f"{key}: {value[:10]}...")
