# server/config.py
import os
from dotenv import load_dotenv

load_dotenv()

# 🔥 IMPORTANT: Prevent HF tokenizer thread issues (Render fix)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class Settings:
    # =========================
    # API KEYS
    # =========================
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

    OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    # =========================
    # MODEL CONFIG
    # =========================
    # MODEL_NAME = os.getenv("MODEL_NAME","stepfun/step-3.5-flash:free")
    MODEL_NAME = os.getenv("MODEL_NAME")
    PDF_PATH = os.getenv("PDF_PATH")
    
    # Embedding configuration
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

    EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE","cpu")

    # =========================
    # RETRIEVAL CONFIG
    # =========================
    RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K"))
    RAG_SCORE_THRESHOLD = float(os.getenv("RAG_SCORE_THRESHOLD"))
    
    # =========================
    # ASSISTANT CONFIG
    # =========================
    ASSISTANT_NAME = os.getenv("ASSISTANT_NAME","Alex AI")
    
    
    # =========================
    # LANGCHAIN (OPTIONAL)
    # =========================
    LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING", "true")
    LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
    LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "resume-assistant")

    # =========================
    # REDIS CONFIG (SAFE)
    # =========================
    REDIS_HOST = os.getenv("REDIS_HOST")
    REDIS_PORT = int(os.getenv("REDIS_PORT"))
    REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")

# =========================
# INSTANCE
# =========================
settings = Settings()

# =========================
# LANGSMITH GLOBALS
# =========================
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "true")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT")
LANGSMITH_ENDPOINT= os.getenv("LANGSMITH_ENDPOINT")
