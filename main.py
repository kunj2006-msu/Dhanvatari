from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import os
import logging
import time
import asyncio

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

# Rate limiting variables
last_api_call_time = 0
MIN_API_INTERVAL = 1.0  # Minimum 1 second between API calls

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
        
        # First, try to list available models
        try:
            logger.info("Listing available models...")
            models = genai.list_models()
            available_models = []
            for m in models:
                if 'generateContent' in m.supported_generation_methods:
                    available_models.append(m.name)
                    logger.info(f"Available model: {m.name}")
            
            if available_models:
                # Try the first available model
                model_name = available_models[0]
                logger.info(f"Attempting to use: {model_name}")
                test_model = genai.GenerativeModel(model_name)
                test_response = test_model.generate_content("Hello")
                gemini_model = test_model
                logger.info(f"SUCCESS: Using model {model_name}")
                logger.info(f"Test response: {test_response.text[:50]}...")
                # Add delay after successful test
                time.sleep(1)
            else:
                logger.error("No models with generateContent support found")
                
        except Exception as list_error:
            logger.warning(f"Could not list models: {list_error}")
            
            # Fallback to models we know are available from the logs
            model_attempts = [
                'models/gemini-2.0-flash',
                'models/gemini-2.0-flash-001',
                'models/gemini-flash-latest',
                'models/gemini-pro-latest',
                'models/gemini-2.5-flash',
                'models/gemini-2.0-flash-lite',
                'models/gemini-2.0-flash-lite-001'
            ]
            
            for model_name in model_attempts:
                try:
                    logger.info(f"Trying model: {model_name}")
                    test_model = genai.GenerativeModel(model_name)
                    test_response = test_model.generate_content("Hello")
                    gemini_model = test_model
                    logger.info(f"SUCCESS: Using model {model_name}")
                    logger.info(f"Test response: {test_response.text[:50]}...")
                    # Add delay between model tests to avoid rate limits
                    time.sleep(1)
                    break
                except Exception as e:
                    logger.warning(f"Model {model_name} failed: {str(e)[:100]}")
                    # Add delay before trying next model
                    time.sleep(1)
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

async def rate_limited_api_call(model, prompt: str, max_retries: int = 3) -> str:
    """Make a rate-limited API call with retry logic for 429 errors"""
    global last_api_call_time
    
    for attempt in range(max_retries):
        try:
            # Ensure minimum time between API calls
            current_time = time.time()
            time_since_last_call = current_time - last_api_call_time
            
            if time_since_last_call < MIN_API_INTERVAL:
                sleep_time = MIN_API_INTERVAL - time_since_last_call
                logger.info(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
                await asyncio.sleep(sleep_time)
            
            # Make the API call
            logger.info(f"Making API call (attempt {attempt + 1}/{max_retries})")
            response = model.generate_content(prompt)
            
            # Update last call time
            last_api_call_time = time.time()
            
            # Add a 1-second pause after successful call
            await asyncio.sleep(1.0)
            
            return response.text
            
        except Exception as e:
            error_str = str(e).lower()
            logger.warning(f"API call attempt {attempt + 1} failed: {str(e)[:100]}")
            
            # Handle specific error types
            if "429" in error_str or "too many requests" in error_str or "quota" in error_str:
                if attempt < max_retries - 1:
                    # Exponential backoff for rate limit errors
                    backoff_time = (2 ** attempt) * 2  # 2, 4, 8 seconds
                    logger.info(f"Rate limit hit, backing off for {backoff_time} seconds")
                    await asyncio.sleep(backoff_time)
                    continue
                else:
                    logger.error("Max retries reached for rate limit error")
                    raise Exception("API rate limit exceeded. Please try again later.")
            
            elif "400" in error_str or "invalid" in error_str:
                # Don't retry for client errors
                logger.error(f"Client error, not retrying: {e}")
                raise e
            
            else:
                # For other errors, wait and retry
                if attempt < max_retries - 1:
                    wait_time = 2 * (attempt + 1)  # 2, 4, 6 seconds
                    logger.info(f"Retrying after {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    logger.error(f"Max retries reached: {e}")
                    raise e
    
    raise Exception("All retry attempts failed")

async def get_ai_response(messages: List[Message], language: str = "English") -> str:
    try:
        if gemini_model:
            logger.info("Using Gemini AI for response with rate limiting")
            
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
            
            # Use rate-limited API call with retry logic
            response_text = await rate_limited_api_call(gemini_model, conversation_text)
            return response_text
        else:
            # Use fallback response
            logger.info("Using fallback response system")
            return FALLBACK_RESPONSES.get(language, FALLBACK_RESPONSES["English"])
            
    except Exception as e:
        logger.error(f"Error in AI response: {e}")
        # Check if it's a rate limit error and provide appropriate message
        error_str = str(e).lower()
        if "rate limit" in error_str or "429" in error_str or "quota" in error_str:
            rate_limit_message = {
                "English": "I apologize, but I'm currently experiencing high demand. Please wait a moment and try again. If this persists, please consult with a healthcare professional directly.",
                "Hindi": "मुझे खेद है, लेकिन मैं वर्तमान में उच्च मांग का सामना कर रहा हूं। कृपया एक क्षण प्रतीक्षा करें और पुनः प्रयास करें। यदि यह जारी रहता है, तो कृपया सीधे किसी स्वास्थ्य पेशेवर से सलाह लें।",
                "Gujarati": "મને માફ કરશો, પરંતુ હું હાલમાં ઉચ્ચ માંગનો સામનો કરી રહ્યો છું. કૃપા કરીને એક ક્ષણ રાહ જુઓ અને ફરીથી પ્રયાસ કરો. જો આ ચાલુ રહે, તો કૃપા કરીને સીધા આરોગ્ય વ્યાવસાયિક સાથે સલાહ લો."
            }
            return rate_limit_message.get(language, rate_limit_message["English"])
        else:
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
