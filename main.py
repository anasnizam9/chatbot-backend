"""
FastAPI Backend for AI-Powered Chatbot
Complete implementation according to project specifications
"""

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.sql import func
from pydantic import BaseModel, EmailStr
from typing import List, Optional
import firebase_admin
from firebase_admin import credentials, auth
import google.generativeai as genai
import os
import json
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable must be set")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Database Models
class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    firebase_uid = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True)
    name = Column(String, nullable=False)
    avatar_url = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    chats = relationship("Chat", back_populates="user", cascade="all, delete-orphan")

class Chat(Base):
    __tablename__ = "chats"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    user = relationship("User", back_populates="chats")
    messages = relationship("Message", back_populates="chat", cascade="all, delete-orphan")

class Message(Base):
    __tablename__ = "messages"
    
    id = Column(Integer, primary_key=True, index=True)
    content = Column(Text, nullable=False)
    role = Column(String, nullable=False)  # "user" or "assistant"
    chat_id = Column(Integer, ForeignKey("chats.id"), nullable=False)
    suggestions = Column(JSON)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    chat = relationship("Chat", back_populates="messages")

# Create tables
Base.metadata.create_all(bind=engine)

# Pydantic Models
class ChatRequest(BaseModel):
    message: str
    chatId: Optional[int] = None

class ChatResponse(BaseModel):
    chatId: int
    response: str
    suggestions: List[str] = []

class UserCreate(BaseModel):
    firebase_uid: str
    email: Optional[EmailStr] = None
    name: str
    avatar_url: Optional[str] = None

class UserResponse(BaseModel):
    id: int
    firebase_uid: str
    email: Optional[str]
    name: str
    avatar_url: Optional[str]
    created_at: datetime
    
    class Config:
        from_attributes = True

# FastAPI app initialization
app = FastAPI(
    title="AI Chatbot API",
    description="FastAPI backend for AI-powered chatbot with Firebase authentication and Gemini AI",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Firebase Admin
try:
    if not firebase_admin._apps:
        # Initialize Firebase Admin with default credentials
        # In production, use service account key file
        firebase_admin.initialize_app()
        print("Firebase Admin initialized successfully")
except Exception as e:
    print(f"Firebase Admin initialization warning: {e}")

# Initialize Gemini AI
try:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    print("Gemini AI configured successfully")
except Exception as e:
    print(f"Gemini AI configuration error: {e}")

# Security
security = HTTPBearer(auto_error=False)

# Dependency functions
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> Optional[User]:
    """Verify Firebase JWT token and get/create user"""
    if not credentials:
        # Return demo user for development
        demo_user = db.query(User).filter(User.firebase_uid == "demo_user").first()
        if not demo_user:
            demo_user = User(
                firebase_uid="demo_user",
                email="demo@example.com",
                name="Demo User"
            )
            db.add(demo_user)
            db.commit()
            db.refresh(demo_user)
        return demo_user
    
    try:
        # Verify Firebase token
        decoded_token = auth.verify_id_token(credentials.credentials)
        firebase_uid = decoded_token['uid']
        
        # Get or create user
        user = db.query(User).filter(User.firebase_uid == firebase_uid).first()
        if not user:
            user = User(
                firebase_uid=firebase_uid,
                email=decoded_token.get('email'),
                name=decoded_token.get('name', 'Anonymous'),
                avatar_url=decoded_token.get('picture')
            )
            db.add(user)
            db.commit()
            db.refresh(user)
        
        return user
    except Exception as e:
        print(f"Auth error: {e}")
        # Return demo user as fallback
        demo_user = db.query(User).filter(User.firebase_uid == "demo_user").first()
        if not demo_user:
            demo_user = User(
                firebase_uid="demo_user",
                email="demo@example.com", 
                name="Demo User"
            )
            db.add(demo_user)
            db.commit()
            db.refresh(demo_user)
        return demo_user

# API Routes
@app.get("/")
async def root():
    return {
        "message": "AI Chatbot FastAPI Backend",
        "status": "running",
        "framework": "FastAPI",
        "ai_model": "Gemini 2.5 Flash"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Main chat endpoint with Gemini AI integration"""
    try:
        # Get or create chat
        if request.chatId:
            chat = db.query(Chat).filter(
                Chat.id == request.chatId,
                Chat.user_id == current_user.id
            ).first()
            if not chat:
                raise HTTPException(status_code=404, detail="Chat not found")
        else:
            # Create new chat
            chat_title = await generate_chat_title(request.message)
            chat = Chat(
                title=chat_title,
                user_id=current_user.id
            )
            db.add(chat)
            db.commit()
            db.refresh(chat)
        
        # Save user message
        user_message = Message(
            content=request.message,
            role="user",
            chat_id=chat.id
        )
        db.add(user_message)
        db.commit()
        
        # Generate AI response
        ai_response = await generate_ai_response(request.message)
        
        # Save AI message
        ai_message = Message(
            content=ai_response["response"],
            role="assistant",
            chat_id=chat.id,
            suggestions=ai_response.get("suggestions", [])
        )
        db.add(ai_message)
        db.commit()
        
        return ChatResponse(
            chatId=chat.id,
            response=ai_response["response"],
            suggestions=ai_response.get("suggestions", [])
        )
        
    except Exception as e:
        print(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail="Failed to process chat message")

@app.get("/api/chats")
async def get_user_chats(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get all chats for current user"""
    chats = db.query(Chat).filter(Chat.user_id == current_user.id).order_by(Chat.updated_at.desc()).all()
    return chats

@app.get("/api/chats/{chat_id}/messages")
async def get_chat_messages(
    chat_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get messages for a specific chat"""
    chat = db.query(Chat).filter(
        Chat.id == chat_id,
        Chat.user_id == current_user.id
    ).first()
    
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    
    messages = db.query(Message).filter(Message.chat_id == chat_id).order_by(Message.created_at).all()
    return messages

@app.delete("/api/chats/{chat_id}")
async def delete_chat(
    chat_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Delete a chat and all messages"""
    chat = db.query(Chat).filter(
        Chat.id == chat_id,
        Chat.user_id == current_user.id
    ).first()
    
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    
    # Delete messages first
    db.query(Message).filter(Message.chat_id == chat_id).delete()
    db.delete(chat)
    db.commit()
    
    return {"message": "Chat deleted successfully"}

@app.post("/api/users", response_model=UserResponse)
async def create_user(user_data: UserCreate, db: Session = Depends(get_db)):
    """Create or get existing user"""
    existing_user = db.query(User).filter(User.firebase_uid == user_data.firebase_uid).first()
    if existing_user:
        return existing_user
    
    user = User(**user_data.dict())
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

# AI Helper Functions
async def generate_ai_response(message: str) -> dict:
    """Generate AI response using Gemini API"""
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        
        prompt = f"""You are a helpful AI assistant. Respond to the user's message in a conversational and informative way.
        After your response, suggest 2-3 follow-up questions or topics the user might be interested in.
        
        Format your response as JSON:
        {{
            "response": "Your detailed response here",
            "suggestions": ["Follow-up question 1", "Follow-up question 2", "Follow-up question 3"]
        }}
        
        User message: {message}"""
        
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.7,
                max_output_tokens=1000,
            )
        )
        
        # Parse JSON response
        try:
            # Clean response text
            text = response.text.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
            
            result = json.loads(text)
            return result
        except json.JSONDecodeError:
            return {
                "response": response.text,
                "suggestions": ["Tell me more", "What else?", "Can you explain that?"]
            }
            
    except Exception as e:
        print(f"AI generation error: {e}")
        return {
            "response": "I apologize, but I'm having trouble generating a response right now. Please try again.",
            "suggestions": ["Try asking something else", "Rephrase your question", "Ask me about something different"]
        }

async def generate_chat_title(message: str) -> str:
    """Generate chat title from first message"""
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        prompt = f"Generate a short, concise title (5 words max) for a chat that starts with: {message}"
        
        response = model.generate_content(prompt)
        title = response.text.strip().replace('"', '')
        return title[:50]
    except:
        return f"Chat about {message[:30]}..."

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)