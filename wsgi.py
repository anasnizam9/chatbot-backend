# PythonAnywhere WSGI configuration
import sys
import os

# Add your project directory to sys.path
sys.path.insert(0, '/home/yourusername/ai-chatbot-backend')

# Set environment variables
os.environ['DATABASE_URL'] = 'your_database_url_here'
os.environ['GEMINI_API_KEY'] = 'your_gemini_api_key_here'

from main import app
application = app
