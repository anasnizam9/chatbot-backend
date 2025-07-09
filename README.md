# FastAPI Backend - AI Chatbot

FastAPI backend service for the AI-powered chatbot application with Google Gemini AI integration, Firebase authentication, and PostgreSQL database.

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export DATABASE_URL="postgresql://username:password@host:port/database"
export GEMINI_API_KEY="your_gemini_api_key"

# Start server
python main.py
# or
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## 📡 API Endpoints

### Core Routes
- `GET /` - API information and status
- `GET /health` - Health check endpoint
- `GET /docs` - Interactive API documentation (Swagger UI)
- `GET /redoc` - Alternative API documentation

### Authentication
- Firebase JWT token authentication
- Automatic user creation/sync with Firebase UID
- Demo user fallback for development

### Chat Operations
```
POST /api/chat
{
  "message": "User message here",
  "chatId": 123  // optional, creates new chat if not provided
}

Response:
{
  "chatId": 123,
  "response": "AI generated response",
  "suggestions": ["Follow-up 1", "Follow-up 2", "Follow-up 3"]
}
```

### User Management
- `POST /api/users` - Create or get user profile
- `GET /api/chats` - Get user's chat history
- `GET /api/chats/{chat_id}/messages` - Get chat messages
- `DELETE /api/chats/{chat_id}` - Delete chat and messages

## 🗄️ Database Schema

### Users Table
```sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    firebase_uid VARCHAR UNIQUE NOT NULL,
    email VARCHAR UNIQUE,
    name VARCHAR NOT NULL,
    avatar_url VARCHAR,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

### Chats Table
```sql
CREATE TABLE chats (
    id SERIAL PRIMARY KEY,
    title VARCHAR NOT NULL,
    user_id INTEGER REFERENCES users(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

### Messages Table
```sql
CREATE TABLE messages (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    role VARCHAR NOT NULL,  -- 'user' or 'assistant'
    chat_id INTEGER REFERENCES chats(id),
    suggestions JSON,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

## 🤖 AI Integration

### Google Gemini AI
- Model: `gemini-2.5-flash`
- Structured JSON responses
- Temperature: 0.7
- Max tokens: 1000

### Response Format
```python
{
    "response": "Detailed AI response",
    "suggestions": ["Question 1", "Question 2", "Question 3"]
}
```

## 🔐 Authentication

### Firebase Admin
- JWT token verification
- User profile sync
- Auto-registration from Firebase

### Security Headers
- CORS configured for frontend
- Bearer token authentication
- Environment-based secrets

## 🌍 Environment Variables

```env
# Required
DATABASE_URL=postgresql://user:password@host:port/db
GEMINI_API_KEY=your_gemini_api_key

# Optional
GOOGLE_APPLICATION_CREDENTIALS=path/to/service-account.json
DEBUG=True
HOST=0.0.0.0
PORT=8000
```

## 🚀 Deployment

### Render
```bash
# Install dependencies
pip install -r requirements.txt

# Start with Gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker
```

### PythonAnywhere
```python
# WSGI configuration
import sys
sys.path.append('/home/yourusername/ai-chatbot/fastapi_backend')

from main import app
application = app
```

## 🧪 Testing

### Manual Testing
```bash
# Health check
curl http://localhost:8000/health

# Chat endpoint
curl -X POST "http://localhost:8000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello, how are you?"}'
```

### API Documentation
Visit `http://localhost:8000/docs` for interactive API testing.

## 📝 Dependencies

- **FastAPI**: Web framework
- **Uvicorn**: ASGI server
- **SQLAlchemy**: Database ORM
- **Pydantic**: Data validation
- **Firebase Admin**: Authentication
- **Google GenerativeAI**: AI responses
- **psycopg2-binary**: PostgreSQL driver

## 🐛 Troubleshooting

### Common Issues
1. **Import errors**: Check Python path and dependencies
2. **Database connection**: Verify DATABASE_URL format
3. **Firebase auth**: Check service account configuration
4. **Gemini API**: Verify API key and quota

### Debug Mode
Set `DEBUG=True` in environment for detailed error messages.

## 📊 Performance

- Async/await for concurrent requests
- Connection pooling for database
- Efficient JSON serialization
- Structured logging

## 🔧 Configuration

### CORS Settings
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Database Settings
- Auto-creates tables on startup
- SQLAlchemy session management
- Connection pooling enabled

This FastAPI backend provides a robust, scalable foundation for the AI chatbot application with comprehensive API documentation and production-ready deployment options.