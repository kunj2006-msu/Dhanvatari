from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import logging
import requests

# --- Configuration ---
HF_TOKEN = os.getenv("HF_TOKEN")
# Use the new Hugging Face router endpoint for OpenAI-compatible chat completions
API_URL = "https://router.huggingface.co/v1/chat/completions"

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Startup Check for API Token ---
if not HF_TOKEN:
    logger.error("FATAL ERROR: HF_TOKEN environment variable not set. The application cannot start without it.")
    raise ValueError("HF_TOKEN is not set, please check your environment variables.")


# --- FastAPI App Initialization ---
app = FastAPI(title="Dhanvantari API - Hugging Face Edition", description="AI-Driven Healthcare Backend with BioMistral", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# --- Pydantic Models ---
class Message(BaseModel):
    role: str
    content: str

class UserContext(BaseModel):
    age: Optional[str] = None
    gender: Optional[str] = None
    conditions: List[str] = []

class ChatRequest(BaseModel):
    messages: List[Message]
    language: str = "English"
    user_context: Optional[UserContext] = None

class ChatResponse(BaseModel):
    response: str

# --- API Endpoints ---
@app.get("/")
async def root():
    return {"message": "Dhanvantari API is running!", "status": "healthy"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "hugging_face_token_present": bool(HF_TOKEN),
        "version": "2.0.0"
    }

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

def build_personalized_system_prompt(language: str, user_context: Optional[UserContext] = None) -> str:
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
        if user_context.conditions and user_context.conditions != ['None']:
            conditions_str = ', '.join(user_context.conditions)
            context_info += f"- Pre-existing Conditions: {conditions_str}\n"
            serious_conditions = ['Diabetes', 'High Blood Pressure', 'Heart Disease', 'Cancer', 'Kidney Disease', 'Liver Disease']
            user_serious_conditions = [c for c in user_context.conditions if c in serious_conditions]
            if user_serious_conditions:
                context_info += f"\n**IMPORTANT:** This user has serious pre-existing conditions ({', '.join(user_serious_conditions)}). Be EXTRA cautious and strongly emphasize the need for immediate medical consultation for any concerning symptoms.\n"
        else:
            context_info += "- Pre-existing Conditions: None reported\n"
        base_prompt += context_info
    return base_prompt

def query_huggingface_api(messages: List[dict]) -> Optional[str]:
    """Calls the Hugging Face OpenAI-compatible chat completions endpoint."""
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "messages": messages,
        "max_tokens": 500,
    }
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            if result.get('choices') and len(result['choices']) > 0:
                content = result['choices'][0].get('message', {}).get('content')
                if content:
                    logger.info("Successfully received response from Hugging Face API.")
                    return content.strip()
                else:
                    logger.error(f"Hugging Face API returned no content in choice: {result}")
                    return None
            else:
                logger.error(f"Hugging Face API returned unexpected format or empty choices: {result}")
                return None
        else:
            logger.error(f"Hugging Face API request failed with status {response.status_code}: {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling Hugging Face API: {e}")
        return None

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        logger.info(f"Received chat request with {len(request.messages)} messages in {request.language}")
        if not request.messages:
            raise HTTPException(status_code=400, detail="No messages provided")

        # Build the messages payload for the OpenAI-compatible endpoint
        system_prompt_content = build_personalized_system_prompt(request.language, request.user_context)
        
        messages_payload = [
            {"role": "system", "content": system_prompt_content}
        ]
        # Add user messages
        for msg in request.messages:
            messages_payload.append({"role": msg.role, "content": msg.content})

        ai_response = query_huggingface_api(messages_payload)

        if ai_response:
            logger.info(f"Generated response: {ai_response[:100]}...")
            return ChatResponse(response=ai_response)
        else:
            logger.error("AI response is None, raising 500 error.")
            raise HTTPException(status_code=500, detail="The AI model failed to generate a response. Please try again later.")

    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
