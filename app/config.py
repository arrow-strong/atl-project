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

ANALYZER_MODEL = "groq-medium"

KB_SIM_THRESHOLD = 0.35

TOP_K = 4
MAX_CONTEXT_CHUNKS = 5

TIER_ORDER = [
    "local-small",
    "groq-medium",
    "groq-large",
]


NLI_MODEL = "cross-encoder/nli-deberta-v3-base"
SUPPORT_THRESHOLD = 0.5
PASS_RATIO = 0.7
MAX_RETRIES = 2

CONF_WEIGHTS = {
    "support": 0.5,
    "retrieval": 0.3,
    "agreement": 0.2,
}

ENABLE_AGREEMENT = True
UNGROUNDED_CAP = 0.6