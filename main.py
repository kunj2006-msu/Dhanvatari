from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import logging
import requests
import random

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

class Doctor(BaseModel):
    name: str
    specialty: str
    city: str
    state: str
    contact: str

class MentalHealthChatRequest(ChatRequest):
    mood: str # e.g., '😡', '😔', '😐', '😊', '😰'


class DoctorFilterOptions(BaseModel):
    states: List[str]
    cities: List[str]
    specialties: List[str]

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
        "Hindi": "आप एक अनुभवी डॉक्टर हैं। शुद्ध और सरल हिंदी में बात करें। किताबी भाषा के बजाय बोलचाल की भाषा का प्रयोग करें।",
        "Gujarati": "તમે એક અનુભવી ડોક્ટર છો. સાદી અને શુદ્ધ ગુજરાતીમાં વાત કરો. અંગ્રેજીનું સીધું ભાષાંતર કરવાનું ટાળો.",
        "Marathi": "तुम्ही एक अनुभवी डॉक्टर आहात. साध्या आणि नैसर्गिक मराठीत बोला. शब्दांचे थेट भाषांतर करू नका.",
        "Bengali": "আপনি একজন অভিজ্ঞ ডাক্তার। সহজ এবং স্বাভাবিক বাংলায় কথা বলুন। আক্ষরিক অনুবাদ করবেন না।",
        "Telugu": "మీరు అనుభవజ్ఞుడైన డాక్టర్. సహజమైన తెలుగులో మాట్లాడండి. నేరుగా అనువదించవద్దు.",
        "Tamil": "நீங்கள் அனுபவம் வாய்ந்த மருத்துவர். இயல்பான தமிழில் பேசுங்கள். அப்படியே மொழிபெயர்க்க வேண்டாம்.",
        "Malayalam": "നിങ്ങൾ പരിചയസമ്പന്നനായ ഡോക്ടറാണ്. ലളിതമായ മലയാളത്തിൽ സംസാരിക്കുക. നേരിട്ട് തർജ്ജിമ ചെയ്യരുത്.",
        "Kannada": "ನೀವು ಅನುಭವಿ ವೈದ್ಯರು. ಸರಳ ಮತ್ತು ನೈಸರ್ಗಿಕ ಕನ್ನಡದಲ್ಲಿ ಮಾತನಾಡಿ. ನೇರವಾಗಿ ಭಾಷಾಂತರಿಸಬೇಡಿ.",
        "Urdu": "آپ ایک تجربہ کار ڈاکٹر ہیں۔ سادہ اور فطری اردو میں بات کریں۔ لفظی ترجمہ سے پرہیز کریں۔",
        "Odia": "ଆପଣ ଜଣେ ଅଭିଜ୍ଞ ଡାକ୍તର। ସରଳ ଓ ପ୍ରାକୃତିક ଓଡ଼ିଆରେ କଥା ହୁଅନ୍ତୁ।",
        "English": "You are an experienced doctor. Speak in clear, professional, and natural English."
    }
    return instructions.get(language, "Speak naturally and fluently in the requested language.")

# --- Facility 4: Dynamic Doctor Directory ---
def generate_doctors_db(num_doctors=500):
    first_names = ["Aarav", "Vivaan", "Aditya", "Vihaan", "Arjun", "Sai", "Reyansh", "Ayaan", "Krishna", "Ishaan", "Ananya", "Diya", "Saanvi", "Aadhya", "Myra", "Aarohi", "Pari", "Riya", "Siya", "Navya"]
    last_names = ["Sharma", "Verma", "Gupta", "Singh", "Patel", "Kumar", "Das", "Mehta", "Shah", "Joshi"]
    specialties = ["Cardiologist", "Dermatologist", "Orthopedic Surgeon", "Pediatrician", "Neurologist", "Gynecologist", "Oncologist", "Psychiatrist", "Radiologist", "General Physician"]
    locations = [
        {"city": "Mumbai", "state": "Maharashtra"}, {"city": "Pune", "state": "Maharashtra"},
        {"city": "Delhi", "state": "Delhi"},
        {"city": "Bengaluru", "state": "Karnataka"}, {"city": "Mysuru", "state": "Karnataka"},
        {"city": "Chennai", "state": "Tamil Nadu"},
        {"city": "Ahmedabad", "state": "Gujarat"}, {"city": "Vadodara", "state": "Gujarat"}, {"city": "Surat", "state": "Gujarat"}, {"city": "Godhra", "state": "Gujarat"},
        {"city": "Kolkata", "state": "West Bengal"},
    ]
    db = []
    for _ in range(num_doctors):
        location = random.choice(locations)
        doctor = {
            "name": f"Dr. {random.choice(first_names)} {random.choice(last_names)}",
            "specialty": random.choice(specialties),
            "city": location["city"],
            "state": location["state"],
            "contact": f"+91 {random.randint(90000, 99999)} {random.randint(10000, 99999)}"
        }
        db.append(doctor)
    return db

doctors_db = generate_doctors_db()

@app.get("/doctors", response_model=List[Doctor])
async def get_doctors(state: Optional[str] = None, city: Optional[str] = None, specialty: Optional[str] = None):
    filtered_doctors = doctors_db
    if state:
        filtered_doctors = [d for d in filtered_doctors if d['state'] == state]
    if city:
        filtered_doctors = [d for d in filtered_doctors if d['city'] == city]
    if specialty:
        filtered_doctors = [d for d in filtered_doctors if d['specialty'] == specialty]
    return filtered_doctors

@app.get("/doctors/options", response_model=DoctorFilterOptions)
async def get_doctor_options():
    """Provides a list of unique states, cities, and specialties for dropdown filters."""
    states = sorted(list(set(d['state'] for d in doctors_db)))
    cities = sorted(list(set(d['city'] for d in doctors_db)))
    specialties = sorted(list(set(d['specialty'] for d in doctors_db)))
    return {"states": states, "cities": cities, "specialties": specialties}

def build_personalized_system_prompt(language: str, user_context: Optional[UserContext] = None) -> str:
    # Check for serious conditions
    has_serious_condition = user_context and user_context.conditions and any(c in ['Diabetes', 'Heart Disease', 'High Blood Pressure'] for c in user_context.conditions)
    
    # If it's a non-English language AND there is a serious condition, we make the prompt 'Loud'
    if language != "English" and has_serious_condition:
        instruction_style = f"URGENT: The user has {', '.join(user_context.conditions)}. Do NOT give a standard reply. You MUST explain how their symptoms interact with these specific conditions in {language}."
    else:
        instruction_style = f"Provide personalized advice based on the user's profile."

    base_prompt = f"""You are Dhanvantari, a professional Vaidya.
{get_language_instruction(language)}

CORE TASK:
{instruction_style}

STRICT RULES:
1. Always prioritize the User's specific health conditions over general advice.
2. Use natural, spoken {language}.
3. Safety: End by recommending a real doctor.
4.Keep your response concise and limited to 5 main points so the user can follow easily.

USER PROFILE:
- Age: {user_context.age if user_context else 'N/A'}
- Conditions: {user_context.conditions if user_context else 'None'}
"""
    return base_prompt

def query_huggingface_api(messages: List[dict]) -> Optional[str]:
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        # Llama 3.3 70B એ અત્યારે ફ્રીમાં ઉપલબ્ધ શ્રેષ્ઠ મોડેલ છે
        "model": "meta-llama/Llama-3.3-70B-Instruct", 
        "messages": messages,
        "max_tokens": 1200,
        "temperature": 0.4,       # Adjusted for linguistic stability
        "frequency_penalty": 0.4,  # પુનરાવર્તન અટકાવવા માટે
        "top_p": 0.85,
        "presence_penalty": 0.3
    }
    # ... બાકીનું લોજિક ...
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

# --- Facility 2: Mental Health --- 
@app.post("/chat/mental-health", response_model=ChatResponse)
async def mental_health_chat(request: MentalHealthChatRequest):
    logger.info(f"Received mental health chat request with mood: {request.mood}")

    mood_interpretation = {
        '😡': 'feeling very angry or frustrated',
        '😔': 'feeling sad or down',
        '😐': 'feeling neutral or unsure',
        '😊': 'feeling happy or positive',
        '😰': 'feeling anxious or scared'
    }

    system_prompt = f"""You are a Compassionate Psychological Counselor. Your goal is to provide a safe, empathetic, and supportive space. The user is currently {mood_interpretation.get(request.mood, 'expressing a certain mood')}. 
    {get_language_instruction(request.language)}

    Your Instructions:
    1. Acknowledge their stated mood gently.
    2. NEVER diagnose. Offer general coping strategies or reflective questions.
    3. Use a soft, reassuring, and non-judgmental tone.
    4. Always encourage speaking to a licensed therapist for serious issues.
    5. Keep the conversation focused on emotional well-being.
    """
    
    messages_payload = [{"role": "system", "content": system_prompt}]
    for msg in request.messages:
        messages_payload.append({"role": msg.role, "content": msg.content})

    ai_response = query_huggingface_api(messages_payload)

    if ai_response:
        return ChatResponse(response=ai_response)
    else:
        raise HTTPException(status_code=500, detail="AI model failed to generate a response.")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
