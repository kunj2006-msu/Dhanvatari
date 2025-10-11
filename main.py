from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import os
import logging

# Simple fallback responses for when Gemini is not available
FALLBACK_RESPONSES = {
    "English": "Thank you for sharing your symptoms. While I'd like to provide personalized guidance, I'm currently experiencing technical difficulties with my AI analysis system. For your health and safety, I strongly recommend consulting with a healthcare professional who can properly evaluate your symptoms and provide appropriate medical advice. If you're experiencing severe symptoms, please seek immediate medical attention.",
    "Hindi": "आपके लक्षण साझा करने के लिए धन्यवाद। जबकि मैं व्यक्तिगत मार्गदर्शन प्रदान करना चाहूंगा, मैं वर्तमान में अपने AI विश्लेषण सिस्टम के साथ तकनीकी कठिनाइयों का सामना कर रहा हूं। आपके स्वास्थ्य और सुरक्षा के लिए, मैं दृढ़ता से सुझाव देता हूं कि किसी स्वास्थ्य पेशेवर से सलाह लें जो आपके लक्षणों का उचित मूल्यांकन कर सके और उचित चिकित्सा सलाह प्रदान कर सके।",
    "Gujarati": "તમારા લક્ષણો શેર કરવા બદલ આભાર. જ્યારે હું વ્યક્તિગત માર્ગદર્શન આપવા માંગુ છું, હું હાલમાં મારી AI વિશ્લેષણ સિસ્ટમ સાથે તકનીકી મુશ્કેલીઓનો સામનો કરી રહ્યો છું. તમારા સ્વાસ્થ્ય અને સલામતી માટે, હું ભારપૂર્વક સૂચન કરું છું કે આરોગ્ય વ્યાવસાયિક સાથે સલાહ લો જે તમારા લક્ષણોનું યોગ્ય મૂલ્યાંકન કરી શકે અને યોગ્ય તબીબી સલાહ આપી શકે."
}

app = FastAPI(title="Dhanvantari API", description="AI-Driven Healthcare Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    language: str = "English"

class ChatResponse(BaseModel):
    response: str

# Try to initialize Gemini
gemini_model = None
try:
    import google.generativeai as genai
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        logger.info(f"API Key found: {api_key[:10]}...")
        genai.configure(api_key=api_key)
        
        # Try simple model names first
        model_attempts = [
            'gemini-1.5-flash-latest',
            'gemini-1.5-pro-latest', 
            'gemini-1.0-pro-latest',
            'gemini-pro',
            'gemini-1.5-flash',
            'gemini-1.5-pro'
        ]
        
        gemini_model = None
        for model_name in model_attempts:
            try:
                logger.info(f"Trying model: {model_name}")
                test_model = genai.GenerativeModel(model_name)
                test_response = test_model.generate_content("Hello")
                gemini_model = test_model
                logger.info(f"SUCCESS: Using model {model_name}")
                logger.info(f"Test response: {test_response.text[:50]}...")
                break
            except Exception as e:
                logger.warning(f"Model {model_name} failed: {str(e)[:100]}")
                continue
        
        if not gemini_model:
            logger.error("All model attempts failed - using fallback system")
    else:
        logger.warning("No GEMINI_API_KEY found")
except Exception as e:
    logger.error(f"Gemini initialization failed: {e}")
    gemini_model = None

@app.get("/")
async def root():
    return {"message": "Dhanvantari API is running!", "status": "healthy"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "gemini_available": gemini_model is not None,
        "api_key_present": bool(os.getenv("GEMINI_API_KEY")),
        "version": "1.0.0"
    }

def get_language_instruction(language: str) -> str:
    if language == "Hindi":
        return "Please respond in Hindi (हिंदी). Use Devanagari script."
    elif language == "Gujarati":
        return "Please respond in Gujarati (ગુજરાતી). Use Gujarati script."
    else:
        return "Please respond in English."

async def get_ai_response(messages: List[Message], language: str = "English") -> str:
    try:
        if gemini_model:
            logger.info("Using Gemini AI for response")
            
            # Build conversation context
            system_prompt = f"""You are Dhanvantari, a helpful AI health companion. 
            {get_language_instruction(language)}
            
            Guidelines:
            - Provide helpful health information
            - Ask clarifying questions if needed
            - Always recommend consulting healthcare professionals
            - Never provide definitive medical diagnosis
            - Be empathetic and caring"""
            
            conversation_text = f"{system_prompt}\n\n"
            for message in messages:
                if message.role == "user":
                    conversation_text += f"User: {message.content}\n"
                elif message.role == "assistant":
                    conversation_text += f"Assistant: {message.content}\n"
            
            conversation_text += "\nPlease provide a helpful, empathetic response following the guidelines above."
            
            response = gemini_model.generate_content(conversation_text)
            return response.text
        else:
            # Use fallback response
            logger.info("Using fallback response system")
            return FALLBACK_RESPONSES.get(language, FALLBACK_RESPONSES["English"])
            
    except Exception as e:
        logger.error(f"Error in AI response: {e}")
        return FALLBACK_RESPONSES.get(language, FALLBACK_RESPONSES["English"])

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        logger.info(f"Received chat request with {len(request.messages)} messages in {request.language}")
        
        if not request.messages:
            raise HTTPException(status_code=400, detail="No messages provided")
        
        ai_response = await get_ai_response(request.messages, request.language)
        logger.info(f"Generated response: {ai_response[:100]}...")
        
        return ChatResponse(response=ai_response)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return ChatResponse(
            response=FALLBACK_RESPONSES.get(request.language, FALLBACK_RESPONSES["English"])
        )

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main_simple:app", host="0.0.0.0", port=port, reload=True)
