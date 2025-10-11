from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import google.generativeai as genai
import os
from dotenv import load_dotenv
import logging

load_dotenv()

app = FastAPI(title="Dhanvantari API", description="AI-Driven Healthcare Backend", version="1.0.0")

# Get allowed origins from environment or use defaults
allowed_origins = os.getenv("ALLOWED_ORIGINS", 
    "http://localhost:3000,http://127.0.0.1:3000,https://dhanvantari-healthcare.netlify.app,https://*.netlify.app"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
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

class ChatResponse(BaseModel):
    response: str

SYSTEM_PROMPT = """You are Dhanvantari, a cautious, empathetic, and helpful AI Health Companion. Your goal is to help users understand their symptoms better, not to diagnose them.

STRICT GUIDELINES:
1. PERSONA: Act as a caring health companion who provides information, not medical diagnosis
2. PROCESS:
   - When a user provides initial symptoms, analyze them and ask 1-2 relevant clarifying questions
   - Once you have enough information, list 2-3 potential conditions that might be related to the symptoms
   - For each condition, provide a simple, one-paragraph explanation in easy-to-understand language
3. CONSTRAINTS (VERY IMPORTANT):
   - NEVER provide a definitive diagnosis
   - Use phrases like "Some conditions that can cause these symptoms include..." or "It might be helpful to read about..."
   - ALWAYS conclude your response by strongly recommending the user consult a real healthcare professional
   - DO NOT ask for personally identifiable information (PII) like name, address, or contact details
   - Keep responses informative but not overly technical
   - Show empathy and understanding
   - If symptoms seem serious or emergency-related, immediately recommend seeking urgent medical care

RESPONSE FORMAT:
- Start with acknowledgment of their concern
- Ask clarifying questions if needed (limit to 1-2 questions)
- Provide potential conditions with simple explanations
- Always end with a strong recommendation to consult healthcare professionals

Remember: You are providing information to help users understand their symptoms better, not replacing professional medical advice."""

def initialize_gemini():
    """Initialize Google Gemini API"""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.warning("GEMINI_API_KEY not found in environment variables")
        return None
    
    # Log the first few characters of the API key for debugging
    logger.info(f"API Key found: {api_key[:10]}...")
    
    try:
        genai.configure(api_key=api_key)
        # Use the correct model name that's available
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Test the API with a simple call
        test_response = model.generate_content("Say hello")
        if test_response and test_response.text:
            logger.info(f"Gemini API initialized successfully with model: gemini-2.0-flash")
            logger.info(f"Test response: {test_response.text}")
            return model
        else:
            logger.warning("Gemini API configured but test failed")
            return None
    except Exception as e:
        logger.error(f"Failed to initialize Gemini API: {e}")
        # Try alternative model
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-pro-latest')
            test_response = model.generate_content("Hello")
            logger.info(f"Gemini API initialized with fallback model: gemini-pro-latest")
            return model
        except Exception as e2:
            logger.error(f"Fallback model also failed: {e2}")
            return None

gemini_model = initialize_gemini()

def analyze_conversation_context(messages: List[Message]) -> dict:
    """Analyze the conversation to understand context and progression"""
    context = {
        "symptoms": [],
        "duration": None,
        "severity": None,
        "additional_info": [],
        "questions_asked": 0,
        "user_responses": 0
    }
    
    all_text = " ".join([msg.content.lower() for msg in messages if msg.role == "user"])
    
    # Extract symptoms
    symptom_keywords = {
        "fever": ["fever", "temperature", "hot", "burning"],
        "cough": ["cough", "coughing", "throat"],
        "headache": ["headache", "head pain", "migraine"],
        "stomach": ["stomach", "nausea", "vomit", "diarrhea", "belly"],
        "cold": ["cold", "runny nose", "sneezing", "congestion"],
        "fatigue": ["tired", "weak", "exhausted", "fatigue"],
        "body_ache": ["body ache", "muscle pain", "joint pain"]
    }
    
    for symptom, keywords in symptom_keywords.items():
        if any(keyword in all_text for keyword in keywords):
            context["symptoms"].append(symptom)
    
    # Extract duration
    if "last night" in all_text or "yesterday" in all_text:
        context["duration"] = "1 day"
    elif "few days" in all_text or "2-3 days" in all_text:
        context["duration"] = "2-3 days"
    elif "week" in all_text:
        context["duration"] = "1 week"
    
    # Count interactions
    context["questions_asked"] = len([msg for msg in messages if msg.role == "assistant" and "?" in msg.content])
    context["user_responses"] = len([msg for msg in messages if msg.role == "user"])
    
    return context

def get_interactive_response(messages: List[Message]) -> str:
    """Generate contextual, interactive responses based on conversation history"""
    if not messages:
        return "Hello! I'm here to help you understand your symptoms. Please tell me what you're experiencing."
    
    context = analyze_conversation_context(messages)
    latest_message = messages[-1].content.lower()
    
    # Multi-turn conversation logic
    if context["user_responses"] == 1:
        # First interaction - gather initial info and ask clarifying questions
        return get_initial_assessment(context, latest_message)
    elif context["user_responses"] == 2:
        # Second interaction - provide analysis and practical advice
        return get_detailed_analysis(context, latest_message)
    else:
        # Ongoing conversation - provide specific guidance
        return get_ongoing_guidance(context, latest_message)

def get_initial_assessment(context: dict, user_message: str) -> str:
    """First response - acknowledge symptoms and ask key questions"""
    symptoms = context["symptoms"]
    
    if "fever" in symptoms and "cough" in symptoms:
        return f"""I see you're dealing with fever and cough - that's concerning and I want to help you understand what this might indicate.

To give you the best guidance, I need to know:
1. What's your current temperature? (if you've measured it)
2. Is your cough dry or are you bringing up mucus/phlegm?
3. Do you feel chills or body aches along with the fever?

Based on your symptoms of fever + cough since {context.get('duration', 'recently')}, this could indicate a respiratory infection. Let me know your answers and I'll provide specific advice including home remedies and when to see a doctor."""

    elif "fever" in symptoms:
        return """I understand you have a fever - this is your body's way of fighting an infection.

Important questions:
1. How high is your temperature? (Normal is 98.6°F/37°C)
2. How long have you had the fever?
3. Any other symptoms like headache, body aches, or chills?

Please share these details so I can give you specific advice about home care and when to seek medical attention."""

    elif "headache" in symptoms:
        return """I understand you're experiencing headaches. Let me help you with this.

To provide targeted advice:
1. Where exactly is the pain? (forehead, temples, back of head, or all over?)
2. Is it throbbing, sharp, or a dull ache?
3. Any triggers you've noticed? (stress, screen time, certain foods?)

Tell me more and I'll suggest specific remedies and prevention strategies."""

    elif "cough" in symptoms:
        return """A persistent cough can be quite bothersome. Let me help you understand it better.

Key questions:
1. Is it a dry cough or are you coughing up mucus?
2. When is it worse? (night, morning, or throughout the day?)
3. Any chest tightness or shortness of breath?

Share these details and I'll provide targeted advice including natural remedies and when to be concerned."""

    else:
        return f"""Thank you for sharing your symptoms with me. I want to help you understand what you're experiencing.

To provide the most helpful guidance:
1. How long have you been experiencing these symptoms?
2. On a scale of 1-10, how would you rate your discomfort?
3. Have you tried anything for relief so far?

Please provide these details so I can give you specific, practical advice."""

def get_detailed_analysis(context: dict, user_message: str) -> str:
    """Second response - provide detailed analysis with home remedies and advice"""
    symptoms = context["symptoms"]
    
    if "fever" in symptoms and "cough" in symptoms:
        if "high" in user_message or "102" in user_message or "103" in user_message:
            return """Based on your high fever with cough, this suggests a significant respiratory infection that needs attention.

**Immediate Home Care:**
🌡️ **Temperature Management:**
- Take paracetamol/acetaminophen every 6 hours (follow package instructions)
- Cool compresses on forehead and wrists
- Drink plenty of fluids (water, herbal teas, broths)

🍯 **Natural Remedies for Cough:**
- Warm turmeric milk with honey before bed
- Ginger-honey tea (1 tsp fresh ginger + 1 tsp honey in hot water)
- Steam inhalation with eucalyptus oil (5-10 minutes, 2-3 times daily)
- Gargle with warm salt water (1/2 tsp salt in warm water)

**When to See a Doctor URGENTLY:**
- Fever above 103°F (39.4°C)
- Difficulty breathing or chest pain
- Persistent vomiting or severe dehydration
- Symptoms worsening after 3-4 days

**Doctor Type:** Visit a General Physician or Internal Medicine doctor first. They can determine if you need specialist care.

Would you like specific guidance on any of these remedies or have questions about your symptoms?"""

        else:
            return """Based on your fever and cough symptoms, this appears to be a respiratory infection. Here's my comprehensive guidance:

**Home Treatment Plan:**

🌡️ **Fever Management:**
- Monitor temperature every 4-6 hours
- Paracetamol 500mg every 6 hours (adults)
- Stay hydrated: 8-10 glasses of water daily
- Rest in a cool, well-ventilated room

🍵 **Proven Home Remedies:**
- **Turmeric Golden Milk:** 1 tsp turmeric + pinch of black pepper in warm milk before sleep
- **Honey-Ginger Tea:** Fresh ginger slice + 1 tbsp honey in hot water (3 times daily)
- **Steam Therapy:** Inhale steam from hot water with 2-3 drops eucalyptus oil
- **Throat Gargle:** Warm salt water (1/2 tsp salt) every 2-3 hours

**Dietary Advice:**
- Light, easily digestible foods (khichdi, soup, fruits)
- Avoid dairy if cough is mucus-heavy
- Increase Vitamin C (oranges, lemons, amla)

**See a General Physician if:**
- Fever persists beyond 3 days
- Cough becomes severe or bloody
- Breathing difficulties develop

How are you feeling now? Any specific remedy you'd like more details about?"""

    elif "headache" in symptoms:
        return """Based on your headache description, here's targeted relief:

**Immediate Relief:**
💊 **Medication:** Paracetamol 500mg or Ibuprofen 400mg (follow package instructions)
🧊 **Cold/Heat Therapy:** Cold compress for migraines, warm compress for tension headaches
💆 **Pressure Points:** Gentle massage of temples, neck, and scalp

**Natural Remedies:**
- **Peppermint Oil:** Dilute 2-3 drops in coconut oil, apply to temples
- **Ginger Tea:** Fresh ginger in hot water with honey
- **Hydration:** Drink 2-3 glasses of water immediately (dehydration common cause)

**Lifestyle Adjustments:**
- Reduce screen time and bright lights
- Ensure 7-8 hours sleep
- Regular meals (low blood sugar triggers headaches)

**See a Neurologist if:**
- Sudden severe headache unlike any before
- Headache with fever, stiff neck, or vision changes
- Frequent headaches (>3 per week)

**General Physician for:** Regular headache evaluation and initial treatment.

Which type of headache relief would you like to try first?"""

    else:
        return """Based on your symptoms, here's a comprehensive care plan:

**General Wellness Approach:**
- Adequate rest (7-8 hours sleep)
- Stay well-hydrated
- Eat nutritious, light meals
- Monitor your symptoms daily

**When to Consult:**
- **General Physician:** For initial evaluation and common conditions
- **Specialist:** If symptoms persist or worsen after initial treatment

**Home Monitoring:**
Keep track of:
- Symptom severity (1-10 scale)
- Duration and triggers
- What helps or worsens symptoms

Would you like specific advice for any particular aspect of your symptoms?"""

def get_ongoing_guidance(context: dict, user_message: str) -> str:
    """Ongoing conversation - provide follow-up guidance and support"""
    if "better" in user_message or "improving" in user_message:
        return """I'm glad to hear you're feeling better! That's a positive sign that your body is healing.

**Continue Your Recovery:**
- Keep taking the remedies that are helping
- Maintain good hydration and rest
- Gradually return to normal activities
- Monitor for any symptom return

**Complete Recovery Tips:**
- Continue immune-boosting foods (citrus fruits, ginger, garlic)
- Gentle exercise when energy returns
- Ensure full rest until completely well

Is there anything specific about your recovery you'd like guidance on?"""

    elif "worse" in user_message or "not better" in user_message:
        return """I'm concerned that you're not improving. This suggests you may need medical evaluation.

**Immediate Steps:**
1. **See a doctor today** if symptoms are worsening
2. **Monitor closely** for warning signs (high fever, breathing issues, severe pain)
3. **Continue supportive care** while seeking medical help

**Which Doctor to See:**
- **General Physician/Family Doctor:** First point of contact
- **Emergency Room:** If severe symptoms (high fever >103°F, difficulty breathing, severe dehydration)

**Before Doctor Visit:**
- Note all symptoms and their progression
- List all remedies tried
- Prepare questions about treatment options

Would you like help preparing for your doctor visit or have urgent concerns I should address?"""

    else:
        return """Thank you for the additional information. Based on our conversation, here's my updated guidance:

**Key Recommendations:**
- Continue the home remedies we discussed
- Monitor your symptoms closely
- Stay well-hydrated and get adequate rest

**Follow-up Care:**
- Reassess your condition in 24-48 hours
- Seek medical care if symptoms persist or worsen
- Continue proven remedies that provide relief

**Questions for You:**
- How are the home remedies working so far?
- Any new symptoms or concerns?
- Do you need clarification on any treatment recommendations?

I'm here to support your recovery. What specific aspect would you like to discuss further?"""

# End of interactive response system

async def get_ai_response(messages: List[Message]) -> str:
    """Get response from AI model or interactive system"""
    try:
        # Try Gemini API first if available
        if gemini_model:
            try:
                logger.info("Using Gemini AI for response")
                
                # Build conversation context for Gemini
                conversation_text = f"{SYSTEM_PROMPT}\n\n"
                
                for message in messages:
                    if message.role == "user":
                        conversation_text += f"User: {message.content}\n"
                    elif message.role == "assistant":
                        conversation_text += f"Assistant: {message.content}\n"
                
                # Add the latest user message context
                conversation_text += "\nPlease provide a helpful, empathetic response following the guidelines above."
                
                response = gemini_model.generate_content(conversation_text)
                
                if response and response.text:
                    logger.info(f"Gemini response generated successfully: {response.text[:100]}...")
                    return response.text
                else:
                    logger.warning("Empty response from Gemini API")
            except Exception as e:
                logger.error(f"Gemini API error: {e}")
        
        # Fallback to interactive system if Gemini fails
        logger.info("Using interactive response system as fallback")
        return get_interactive_response(messages)
            
    except Exception as e:
        logger.error(f"Error getting AI response: {e}")
        return get_interactive_response(messages)

@app.get("/")
async def root():
    return {"message": "Dhanvantari API is running", "status": "healthy"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "gemini_available": gemini_model is not None,
        "version": "1.0.0"
    }

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Main chat endpoint for processing user messages"""
    try:
        logger.info(f"Received chat request: {request}")
        
        if not request.messages:
            logger.warning("No messages provided in request")
            raise HTTPException(status_code=400, detail="No messages provided")
        
        logger.info(f"Processing chat request with {len(request.messages)} messages")
        for i, msg in enumerate(request.messages):
            logger.info(f"Message {i}: role={msg.role}, content={msg.content[:50]}...")
        
        ai_response = await get_ai_response(request.messages)
        logger.info(f"Generated response: {ai_response[:100]}...")
        
        return ChatResponse(response=ai_response)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        # Return a proper response instead of raising an exception
        return ChatResponse(
            response="I apologize, but I'm experiencing technical difficulties. Please try again in a moment, and if the issue persists, please consult with a healthcare professional directly."
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
