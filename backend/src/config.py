import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Models (as strictly defined by the assignment)
MODEL_8B = "llama-3.1-8b-instant"
MODEL_70B = "llama-3.3-70b-versatile"

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_DIR = os.path.join(os.path.dirname(BASE_DIR), "docs")
CHROMA_DB_DIR = os.path.join(BASE_DIR, "data", "chromadb")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing. Please set it in backend/.env")