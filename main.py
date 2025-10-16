from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import logging
import time
import asyncio

# Simple fallback responses for when Gemini is not available
FALLBACK_RESPONSES = {
    "English": "Thank you for sharing your symptoms. While I'd like to provide personalized guidance, I'm currently experiencing technical difficulties with my AI analysis system. For your health and safety, I strongly recommend consulting with a healthcare professional who can properly evaluate your symptoms and provide appropriate medical advice. If you're experiencing severe symptoms, please seek immediate medical attention.",
    "Hindi": "आपके लक्षण साझा करने के लिए धन्यवाद। जबकि मैं व्यक्तिगत मार्गदर्शन प्रदान करना चाहूंगा, मैं वर्तमान में अपने AI विश्लेषण सिस्टम के साथ तकनीकी कठिनाइयों का सामना कर रहा हूं। आपके स्वास्थ्य और सुरक्षा के लिए, मैं दृढ़ता से सुझाव देता हूं कि किसी स्वास्थ्य पेशेवर से सलाह लें जो आपके लक्षणों का उचित मूल्यांकन कर सके और उचित चिकित्सा सलाह प्रदान कर सके।",
    "Bengali": "আপনার লক্ষণ শেয়ার করার জন্য ধন্যবাদ। যদিও আমি ব্যক্তিগত নির্দেশনা প্রদান করতে চাই, আমি বর্তমানে আমার AI বিশ্লেষণ সিস্টেমের সাথে প্রযুক্তিগত সমস্যার সম্মুখীন হচ্ছি। আপনার স্বাস্থ্য এবং নিরাপত্তার জন্য, আমি দৃঢ়ভাবে সুপারিশ করি যে একজন স্বাস্থ্যসেবা পেশাদারের সাথে পরামর্শ করুন।",
    "Marathi": "तुमची लक्षणे सामायिक केल्याबद्दल धन्यवाद। मी वैयक्तिक मार्गदर्शन प्रदान करू इच्छितो, परंतु सध्या माझ्या AI विश्लेषण प्रणालीमध्ये तांत्रिक अडचणी येत आहेत। तुमच्या आरोग्य आणि सुरक्षिततेसाठी, मी जोरदार शिफारस करतो की आरोग्य व्यावसायिकांशी सल्लामसलत करा।",
    "Telugu": "మీ లక్షణాలను పంచుకున్నందుకు ధన్యవాదాలు. నేను వ్యక్తిగత మార్గదర్శకత్వం అందించాలని అనుకుంటున్నాను, కానీ ప్రస్తుతం నా AI విశ్లేషణ వ్యవస్థతో సాంకేతిక ఇబ్బందులు ఎదుర్కొంటున్నాను. మీ ఆరోగ్యం మరియు భద్రత కోసం, ఆరోగ్య నిపుణులతో సంప్రదించాలని నేను గట్టిగా సిఫార్సు చేస్తున్నాను।",
    "Tamil": "உங்கள் அறிகுறிகளைப் பகிர்ந்ததற்கு நன்றி. நான் தனிப்பட்ட வழிகாட்டுதலை வழங்க விரும்பினாலும், தற்போது எனது AI பகுப்பாய்வு அமைப்பில் தொழில்நுட்ப சிக்கல்களை எதிர்கொண்டுள்ளேன். உங்கள் ஆரோக்கியம் மற்றும் பாதுகாப்பிற்காக, சுகாதார நிபுணர்களுடன் ஆலோசிக்குமாறு நான் கடுமையாக பரிந்துரைக்கிறேன்।",
    "Gujarati": "તમારા લક્ષણો શેર કરવા બદલ આભાર. જ્યારે હું વ્યક્તિગત માર્ગદર્શન આપવા માંગુ છું, હું હાલમાં મારી AI વિશ્લેષણ સિસ્ટમ સાથે તકનીકી મુશ્કેલીઓનો સામનો કરી રહ્યો છું. તમારા સ્વાસ્થ્ય અને સલામતી માટે, હું ભારપૂર્વક સૂચન કરું છું કે આરોગ્ય વ્યાવસાયિક સાથે સલાહ લો જે તમારા લક્ષણોનું યોગ્ય મૂલ્યાંકન કરી શકે અને યોગ્ય તબીબી સલાહ આપી શકે.",
    "Urdu": "اپنی علامات شیئر کرنے کے لیے شکریہ۔ اگرچہ میں ذاتی رہنمائی فراہم کرنا چاہوں گا، لیکن فی الوقت میں اپنے AI تجزیاتی نظام کے ساتھ تکنیکی مشکلات کا سامنا کر رہا ہوں۔ آپ کی صحت اور حفاظت کے لیے، میں بھرپور تجویز کرتا ہوں کہ صحت کے پیشہ ور سے مشورہ کریں۔",
    "Kannada": "ನಿಮ್ಮ ಲಕ್ಷಣಗಳನ್ನು ಹಂಚಿಕೊಂಡಿದ್ದಕ್ಕಾಗಿ ಧನ್ಯವಾದಗಳು. ನಾನು ವೈಯಕ್ತಿಕ ಮಾರ್ಗದರ್ಶನವನ್ನು ಒದಗಿಸಲು ಬಯಸುತ್ತೇನೆ, ಆದರೆ ಪ್ರಸ್ತುತ ನನ್ನ AI ವಿಶ್ಲೇಷಣಾ ವ್ಯವಸ್ಥೆಯೊಂದಿಗೆ ತಾಂತ್ರಿಕ ತೊಂದರೆಗಳನ್ನು ಎದುರಿಸುತ್ತಿದ್ದೇನೆ. ನಿಮ್ಮ ಆರೋಗ್ಯ ಮತ್ತು ಸುರಕ್ಷತೆಗಾಗಿ, ಆರೋಗ್ಯ ವೃತ್ತಿಪರರೊಂದಿಗೆ ಸಮಾಲೋಚಿಸಲು ನಾನು ಬಲವಾಗಿ ಶಿಫಾರಸು ಮಾಡುತ್ತೇನೆ।",
    "Odia": "ଆପଣଙ୍କର ଲକ୍ଷଣ ସାଝା କରିଥିବାରୁ ଧନ୍ୟବାଦ। ଯଦିଓ ମୁଁ ବ୍ୟକ୍ତିଗତ ମାର୍ଗଦର୍ଶନ ପ୍ରଦାନ କରିବାକୁ ଚାହେଁ, ବର୍ତ୍ତମାନ ମୁଁ ମୋର AI ବିଶ୍ଳେଷଣ ସିଷ୍ଟମ ସହିତ ଯାନ୍ତ୍ରିକ ଅସୁବିଧାର ସମ୍ମୁଖୀନ ହେଉଛି। ଆପଣଙ୍କର ସ୍ୱାସ୍ଥ୍ୟ ଏବଂ ନିରାପତ୍ତା ପାଇଁ, ମୁଁ ଦୃଢ଼ଭାବେ ପରାମର୍ଶ ଦେଉଛି ଯେ ସ୍ୱାସ୍ଥ୍ୟ ବିଶେଷଜ୍ଞଙ୍କ ସହିତ ପରାମର୍ଶ କରନ୍ତୁ।",
    "Malayalam": "നിങ്ങളുടെ ലക്ഷണങ്ങൾ പങ്കിട്ടതിന് നന്ദി. വ്യക്തിഗത മാർഗ്ഗനിർദ്ദേശം നൽകാൻ ഞാൻ ആഗ്രഹിക്കുന്നുണ്ടെങ്കിലും, നിലവിൽ എന്റെ AI വിശകലന സംവിധാനത്തിൽ സാങ്കേതിക പ്രശ്നങ്ങൾ നേരിടുന്നു. നിങ്ങളുടെ ആരോഗ്യത്തിനും സുരക്ഷയ്ക്കും വേണ്ടി, ആരോഗ്യ വിദഗ്ധരുമായി കൂടിയാലോചിക്കാൻ ഞാൻ ശക്തമായി ശുപാർശ ചെയ്യുന്നു।"
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

# Intelligent rate limiting variables
last_api_call_time = 0
REQUEST_DELAY = 1.5  # Minimum 1.5 seconds between API calls

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

# Initialize Gemini with robust error handling
gemini_model = None
gemini_available = False

def initialize_gemini():
    """Initialize Gemini with comprehensive error handling"""
    global gemini_model, gemini_available
    
    try:
        import google.generativeai as genai
        api_key = os.getenv("GEMINI_API_KEY")
        
        if not api_key:
            logger.warning("No GEMINI_API_KEY found - running in fallback mode")
            return False
            
        logger.info(f"API Key found: {api_key[:10]}...")
        genai.configure(api_key=api_key)
        
        # Try models in order of preference (most reliable first)
        model_attempts = [
            'models/gemini-2.0-flash',
            'models/gemini-flash-latest', 
            'models/gemini-pro-latest',
            'models/gemini-2.5-flash',
            'models/gemini-2.0-flash-lite'
        ]
        
        for model_name in model_attempts:
            try:
                logger.info(f"Attempting to initialize: {model_name}")
                test_model = genai.GenerativeModel(model_name)
                
                # Simple test without quota-heavy operations
                logger.info(f"Testing model: {model_name}")
                # Skip the test generation to avoid quota issues
                gemini_model = test_model
                gemini_available = True
                logger.info(f"SUCCESS: Gemini model {model_name} initialized")
                return True
                
            except Exception as e:
                error_msg = str(e).lower()
                logger.warning(f"Model {model_name} failed: {str(e)[:100]}")
                
                # If it's a quota error, don't try other models
                if "quota" in error_msg or "429" in error_msg:
                    logger.error("Quota exceeded - running in fallback mode")
                    break
                    
                continue
        
        logger.error("All Gemini models failed - running in fallback mode")
        return False
        
    except ImportError:
        logger.error("google-generativeai not installed - running in fallback mode")
        return False
    except Exception as e:
        logger.error(f"Gemini initialization failed: {e} - running in fallback mode")
        return False

# Initialize Gemini on startup
logger.info("Initializing Gemini AI...")
gemini_success = initialize_gemini()

if gemini_success:
    logger.info("✅ Gemini AI initialized successfully")
else:
    logger.info("⚠️ Running in fallback mode - AI responses will use predefined templates")

@app.get("/")
async def root():
    return {"message": "Dhanvantari API is running!", "status": "healthy"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "gemini_available": gemini_available,
        "api_key_present": bool(os.getenv("GEMINI_API_KEY")),
        "fallback_mode": not gemini_available,
        "version": "1.0.0"
    }

def get_language_instruction(language: str) -> str:
    language_instructions = {
        "Hindi": "Please respond in Hindi (हिंदी). Use Devanagari script.",
        "Bengali": "Please respond in Bengali (বাংলা). Use Bengali script.",
        "Marathi": "Please respond in Marathi (मराठी). Use Devanagari script.",
        "Telugu": "Please respond in Telugu (తెలుగు). Use Telugu script.",
        "Tamil": "Please respond in Tamil (தமிழ்). Use Tamil script.",
        "Gujarati": "Please respond in Gujarati (ગુજરાતી). Use Gujarati script.",
        "Urdu": "Please respond in Urdu (اردو). Use Arabic script.",
        "Kannada": "Please respond in Kannada (ಕನ್ನಡ). Use Kannada script.",
        "Odia": "Please respond in Odia (ଓଡ଼ିଆ). Use Odia script.",
        "Malayalam": "Please respond in Malayalam (മലയാളം). Use Malayalam script."
    }
    return language_instructions.get(language, "Please respond in English.")

async def intelligent_rate_limited_api_call(model, prompt: str, max_retries: int = 3) -> str:
    """Make an intelligent rate-limited API call - first request is instant, subsequent requests are delayed if needed"""
    global last_api_call_time
    
    for attempt in range(max_retries):
        try:
            # Intelligent rate limiting: only delay if there was a recent call
            current_time = time.time()
            
            if last_api_call_time > 0:  # Only apply delay if there was a previous call
                time_since_last_call = current_time - last_api_call_time
                
                if time_since_last_call < REQUEST_DELAY:
                    sleep_time = REQUEST_DELAY - time_since_last_call
                    logger.info(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
                    await asyncio.sleep(sleep_time)
            else:
                logger.info("First API call - no delay applied")
            
            # Make the API call
            logger.info(f"Making API call (attempt {attempt + 1})")
            response = model.generate_content(prompt)
            
            # Update last call time AFTER successful call
            last_api_call_time = time.time()
            
            return response.text
            
        except Exception as e:
            error_str = str(e).lower()
            logger.error(f"API call failed (attempt {attempt + 1}): {e}")
            
            # Handle rate limit errors specifically
            if "429" in error_str or "rate limit" in error_str or "quota" in error_str:
                if attempt < max_retries - 1:
                    # Exponential backoff for rate limit errors
                    wait_time = 2 * (attempt + 1)  # 2, 4, 6 seconds
                    logger.info(f"Rate limit hit - retrying after {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    logger.error(f"Max retries reached: {e}")
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

def build_personalized_system_prompt(language: str, user_context: Optional[UserContext] = None) -> str:
    """Build a personalized system prompt based on user context"""
    base_prompt = f"""You are Dhanvantari, a cautious AI Health Companion. Your primary goal is to provide helpful, safe, and general health information, NOT a medical diagnosis.
    {get_language_instruction(language)}
    
    Your Instructions:
    1. **Acknowledge the Context:** Always consider the user's profile in your responses.
    2. **Be Extra Cautious:** If a user with serious pre-existing conditions reports related symptoms, you must be MORE emphatic about the need to see a doctor immediately.
    3. **Tailor Your Questions:** Your follow-up questions should be more specific based on the user's profile.
    4. **NEVER Diagnose:** Reinforce that you are not a doctor. Use phrases like 'Given your profile, it is especially important to consult a doctor...'
    5. **Always Conclude Safely:** End every response by strongly recommending a consultation with a qualified healthcare professional.
    
    Guidelines:
    - Provide helpful health information
    - Ask clarifying questions if needed
    - Always recommend consulting healthcare professionals
    - Never provide definitive medical diagnosis
    - Be empathetic and caring"""
    
    if user_context:
        context_info = "\n\n**User's Context:**\n"
        if user_context.age:
            context_info += f"- **Age:** {user_context.age}\n"
        if user_context.gender:
            context_info += f"- **Gender:** {user_context.gender}\n"
        if user_context.conditions and user_context.conditions != ['None']:
            conditions_str = ', '.join(user_context.conditions)
            context_info += f"- **Pre-existing Conditions:** {conditions_str}\n"
            
            # Add specific warnings for serious conditions
            serious_conditions = ['Diabetes', 'High Blood Pressure', 'Heart Disease', 'Cancer', 'Kidney Disease', 'Liver Disease']
            user_serious_conditions = [c for c in user_context.conditions if c in serious_conditions]
            if user_serious_conditions:
                context_info += f"\n**IMPORTANT:** This user has serious pre-existing conditions ({', '.join(user_serious_conditions)}). Be EXTRA cautious and strongly emphasize the need for immediate medical consultation for any concerning symptoms.\n"
        else:
            context_info += "- **Pre-existing Conditions:** None reported\n"
            
        base_prompt += context_info
    
    return base_prompt

def get_personalized_fallback_response(language: str, user_context: Optional[UserContext] = None) -> str:
    """Generate a personalized fallback response based on user context"""
    base_response = FALLBACK_RESPONSES.get(language, FALLBACK_RESPONSES["English"])
    
    if user_context:
        # Add personalized elements based on user context
        age_context = ""
        if user_context.age:
            try:
                age_num = int(user_context.age)
                if age_num < 18:
                    age_context = " As a young person, it's especially important to involve a parent or guardian in health decisions."
                elif age_num > 65:
                    age_context = " Given your age, please prioritize regular health check-ups and don't hesitate to seek medical attention."
            except:
                pass
        
        condition_context = ""
        if user_context.conditions and user_context.conditions != ['None']:
            serious_conditions = ['Diabetes', 'High Blood Pressure', 'Heart Disease', 'Cancer', 'Kidney Disease', 'Liver Disease']
            user_serious_conditions = [c for c in user_context.conditions if c in serious_conditions]
            if user_serious_conditions:
                condition_context = f" Given your medical history of {', '.join(user_serious_conditions)}, it's crucial to consult with your healthcare provider promptly."
        
        # Combine base response with personalized context
        personalized_response = base_response + age_context + condition_context
        return personalized_response
    
    return base_response

async def get_ai_response(messages: List[Message], language: str = "English", user_context: Optional[UserContext] = None) -> str:
    try:
        if gemini_available and gemini_model:
            logger.info("Using Gemini AI for response with rate limiting")
            logger.info(f"User context: {user_context}")
            
            # Build personalized system prompt
            system_prompt = build_personalized_system_prompt(language, user_context)
            
            conversation_text = f"{system_prompt}\n\n"
            for message in messages:
                if message.role == "user":
                    conversation_text += f"User: {message.content}\n"
                elif message.role == "assistant":
                    conversation_text += f"Assistant: {message.content}\n"
            
            conversation_text += "\nPlease provide a helpful, empathetic response following the guidelines above."
            
            # Use intelligent rate-limited API call with retry logic
            response_text = await intelligent_rate_limited_api_call(gemini_model, conversation_text)
            return response_text
        else:
            # Use enhanced fallback response with personalization
            logger.info("Using personalized fallback response system")
            return get_personalized_fallback_response(language, user_context)
            
    except Exception as e:
        logger.error(f"Error in AI response: {e}")
        # Always use personalized fallback response for any error
        logger.info("Using personalized fallback response due to error")
        return get_personalized_fallback_response(language, user_context)

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        logger.info(f"Received chat request with {len(request.messages)} messages in {request.language}")
        if request.user_context:
            logger.info(f"User context: age={request.user_context.age}, gender={request.user_context.gender}, conditions={request.user_context.conditions}")
        
        if not request.messages:
            raise HTTPException(status_code=400, detail="No messages provided")
        
        ai_response = await get_ai_response(request.messages, request.language, request.user_context)
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
