import os
import asyncio
import aiohttp
import certifi
import motor.motor_asyncio
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, EmailStr
from typing import List, Optional
from passlib.context import CryptContext
from dotenv import load_dotenv
import logging
import threading
import time
import requests

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
MONGO_DETAILS = os.getenv("MONGO_DETAILS")
if not MONGO_DETAILS:
    logger.error("FATAL ERROR: MONGO_DETAILS environment variable not set.")
    # In a real app, you might exit or raise a more specific startup exception.
    # For now, this will prevent the app from trying to connect to localhost.
    raise ValueError("MONGO_DETAILS is not set, please check your .env file.")
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    logger.error("FATAL ERROR: HF_TOKEN environment variable not set.")
    raise ValueError("HF_TOKEN is not set, please check your environment variables.")
# Use the new Hugging Face router endpoint (old api-inference.huggingface.co is deprecated)
API_URL = "https://router.huggingface.co/models/BioMistral/BioMistral-7B-SLERP"

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Dhanvantari V2 API",
    description="Next-generation AI-Driven Healthcare Backend with BioMistral, MongoDB, and enhanced security.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# --- Database Connection ---
ca = certifi.where()
client = motor.motor_asyncio.AsyncIOMotorClient(
    MONGO_DETAILS,
    tlsCAFile=ca,
    serverSelectionTimeoutMS=20000
)
db = client.dhanvantari_v2
user_collection = db.get_collection("users")

# --- Security ---
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# --- Pydantic Models ---
class UserSchema(BaseModel):
    fullname: str = Field(..., min_length=3)
    email: EmailStr = Field(...)
    password: str = Field(..., min_length=6)
    age: Optional[int] = Field(None, ge=0)
    gender: Optional[str] = Field(None)
    conditions: List[str] = Field([], alias="preExistingConditions")

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "fullname": "John Doe",
                "email": "johndoe@example.com",
                "password": "strongpassword123",
                "age": 30,
                "gender": "Male",
                "preExistingConditions": ["Diabetes", "High Blood Pressure"]
            }
        }

class UserLoginSchema(BaseModel):
    email: EmailStr = Field(...)
    password: str = Field(...)

class Message(BaseModel):
    role: str
    content: str

class UserContext(BaseModel):
    age: Optional[int] = None
    gender: Optional[str] = None
    conditions: List[str] = []

class ChatRequest(BaseModel):
    messages: List[Message]
    language: str = "English"
    user_context: Optional[UserContext] = None

# --- Helper Functions ---
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

# --- Routes ---
@app.get("/")
async def root():
    return {"message": "Welcome to Dhanvantari V2 API"}

@app.post("/register", response_description="Add new user")
async def create_user(user_data: UserSchema = Body(...)):
    existing_user = await user_collection.find_one({"email": user_data.email})
    if existing_user:
        raise HTTPException(status_code=409, detail="User with this email already exists.")

    hashed_password = get_password_hash(user_data.password)
    user_dict = user_data.dict()
    user_dict['password'] = hashed_password
    
    new_user = await user_collection.insert_one(user_dict)
    created_user = await user_collection.find_one({"_id": new_user.inserted_id})
    
    # Don't send password back
    del created_user['password']
    return created_user

def get_language_instruction(language: str) -> str:
    instructions = {
        "Hindi": "Please respond in Hindi (हिंदी).",
        "Bengali": "Please respond in Bengali (বাংলা).",
        "Marathi": "Please respond in Marathi (मराठी).",
        "Telugu": "Please respond in Telugu (తెలుగు).",
        "Tamil": "Please respond in Tamil (தமிழ்).",
        "Gujarati": "Please respond in Gujarati (ગુજરાતી).",
        "Urdu": "Please respond in Urdu (اردو).",
        "Kannada": "Please respond in Kannada (ಕನ್ನಡ).",
        "Odia": "Please respond in Odia (ଓଡ଼ିଆ).",
        "Malayalam": "Please respond in Malayalam (മലയാളം).",
        "English": "Please respond in English."
    }
    return instructions.get(language, instructions["English"])

def build_system_prompt(language: str, user_context: Optional[UserContext] = None) -> str:
    base_prompt = f"""You are Dhanvantari, a cautious AI Health Companion. Your primary goal is to provide helpful, safe, and general health information, NOT a medical diagnosis.
{get_language_instruction(language)}

Your Instructions:
1. Acknowledge the Context: Always consider the user's profile in your responses.
2. Be Extra Cautious: If a user with serious pre-existing conditions reports related symptoms, you must be MORE emphatic about the need to see a doctor immediately.
3. NEVER Diagnose: Reinforce that you are not a doctor. Use phrases like 'Given your profile, it is especially important to consult a doctor...'
4. Always Conclude Safely: End every response by strongly recommending a consultation with a qualified healthcare professional."""

    if user_context:
        context_info = "\n\n**User's Context:**\n"
        if user_context.age:
            context_info += f"- Age: {user_context.age}\n"
        if user_context.gender:
            context_info += f"- Gender: {user_context.gender}\n"
        if user_context.conditions:
            conditions_str = ', '.join(user_context.conditions)
            context_info += f"- Pre-existing Conditions: {conditions_str}\n"
            serious_conditions = ['Diabetes', 'High Blood Pressure', 'Heart Disease']
            if any(c in conditions_str for c in serious_conditions):
                context_info += "\nIMPORTANT: This user has serious pre-existing conditions. Be EXTRA cautious and strongly emphasize the need for immediate medical consultation for any concerning symptoms.\n"
        base_prompt += context_info
    return base_prompt

async def query_biomistral(payload):
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    async with aiohttp.ClientSession() as session:
        async with session.post(API_URL, headers=headers, json=payload) as response:
            if response.status != 200:
                logger.error(f"BioMistral API error: {response.status} {await response.text()}")
                return None
            return await response.json()

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    if not HF_TOKEN:
        raise HTTPException(status_code=500, detail="Hugging Face API token is not configured.")

    system_prompt = build_system_prompt(request.language, request.user_context)
    conversation_history = "".join([f"{m.role}: {m.content}\n" for m in request.messages])
    full_prompt = f"{system_prompt}\n\n{conversation_history}assistant:"

    payload = {
        "inputs": full_prompt,
        "parameters": {
            "max_new_tokens": 250,
            "return_full_text": False,
        }
    }

    try:
        api_response = await query_biomistral(payload)
        if api_response and isinstance(api_response, list) and api_response[0].get('generated_text'):
            ai_message = api_response[0]['generated_text'].strip()
            return {"response": ai_message}
        else:
            logger.error(f"Unexpected API response format: {api_response}")
            raise HTTPException(status_code=500, detail="Failed to get a valid response from the AI model.")
    except Exception as e:
        logger.error(f"Error during chat processing: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while processing your request.")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# --- Heartbeat for Render ---
def heartbeat():
    while True:
        try:
            # Replace with your actual Render URL
            render_url = os.getenv("RENDER_URL", "http://localhost:8000")
            if render_url and render_url != "http://localhost:8000":
                requests.get(f"{render_url}/health")
                logger.info("Heartbeat ping sent.")
        except requests.exceptions.RequestException as e:
            logger.error(f"Heartbeat failed: {e}")
        time.sleep(600)  # 10 minutes

# Start heartbeat thread
heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
heartbeat_thread.start()

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
