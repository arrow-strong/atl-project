import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

# Model registry: name -> provider + model id + tier
MODELS = {
    "local-small": {
        "provider": "ollama",
        "model_id": "llama3.2:3b",
        "tier": "small",
    },
    "groq-medium": {
        "provider": "groq",
        "model_id": "openai/gpt-oss-20b",
        "tier": "medium",
    },
    "groq-large": {
        "provider": "groq",
        "model_id": "openai/gpt-oss-120b",
        "tier": "large",
    },
}
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DOCS_DIR = BASE_DIR / "data" / "docs"
CHROMA_DIR = str(BASE_DIR / "data" / "chroma_db")

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "atl_kb"

CHUNK_WORDS = 350
CHUNK_OVERLAP = 50