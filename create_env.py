import os

# Create .env file with proper content
env_content = """GEMINI_API_KEY=AIzaSyAXhVm_SOMNG-6ZWEqMH7y-62jQQwZ1WZY
DEBUG=True
LOG_LEVEL=INFO"""

# Write to .env file
with open('.env', 'w', encoding='utf-8') as f:
    f.write(env_content)

print("Created .env file successfully")

# Test if it can be read
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
print(f"API Key loaded: {api_key is not None}")
if api_key:
    print(f"API Key starts with: {api_key[:10]}...")
