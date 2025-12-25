from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import logging
import requests

# --- Configuration ---
HF_TOKEN = os.getenv("HF_TOKEN")
API_URL = "https://api-inference.huggingface.co/models/BioMistral/BioMistral-7B-SLERP"

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Startup Check for API Token ---
if not HF_TOKEN:
    logger.error("FATAL ERROR: HF_TOKEN environment variable not set. The application cannot start without it.")
    raise ValueError("HF_TOKEN is not set, please check your environment variables.")

# --- Fallback Responses ---
FALLBACK_RESPONSES = {
    "English": "Thank you for sharing. I am currently experiencing technical difficulties and cannot process your request. For your health and safety, please consult a qualified healthcare professional.",
    "Hindi": "साझा करने के लिए धन्यवाद। मैं वर्तमान में तकनीकी कठिनाइयों का सामना कर रहा हूं और आपके अनुरोध को संसाधित नहीं कर सकता। आपके स्वास्थ्य और सुरक्षा के लिए, कृपया एक योग्य स्वास्थ्य देखभाल पेशेवर से परामर्श लें।",
    # Add other languages as needed
}

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

def query_huggingface_api(prompt: str) -> Optional[str]:
    """Calls the Hugging Face Inference API and returns the generated text."""
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 250,
            "return_full_text": False,
        },
        "options": {
            "wait_for_model": True
        }
    }
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and result and 'generated_text' in result[0]:
                logger.info("Successfully received response from Hugging Face API.")
                return result[0]['generated_text'].strip()
            else:
                logger.error(f"Hugging Face API returned unexpected format: {result}")
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

        system_prompt = build_personalized_system_prompt(request.language, request.user_context)
        conversation_history = "".join([f"{m.role}: {m.content}\n" for m in request.messages])
        full_prompt = f"{system_prompt}\n\n{conversation_history}assistant:"

        ai_response = query_huggingface_api(full_prompt)

        if ai_response:
            logger.info(f"Generated response: {ai_response[:100]}...")
            return ChatResponse(response=ai_response)
        else:
            logger.error("AI response is None, returning fallback.")
            fallback_response = FALLBACK_RESPONSES.get(request.language, FALLBACK_RESPONSES["English"])
            return ChatResponse(response=fallback_response)

    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        fallback_response = FALLBACK_RESPONSES.get(request.language, FALLBACK_RESPONSES["English"])
        raise HTTPException(status_code=500, detail=fallback_response)

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
