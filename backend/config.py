from pathlib import Path
import os


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data_sv1"

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

EMBED_MODEL = os.getenv(
    "EMBED_MODEL",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
)
RERANKER_MODEL = os.getenv(
    "RERANKER_MODEL",
    "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
)
ENABLE_RERANKER = os.getenv("ENABLE_RERANKER", "true").lower() == "true"
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "12"))
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "5"))
RERANKER_LOCAL_ONLY = os.getenv("RERANKER_LOCAL_ONLY", "true").lower() == "true"

QDRANT_PATH = Path(os.getenv("QDRANT_PATH", str(DATA_DIR / "qdrant_local")))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "dut_chunks")
MAX_CTX_CHARS = int(os.getenv("MAX_CTX_CHARS", "3000"))
TOP_K = int(os.getenv("TOP_K", "5"))
MAX_GENERATION_RETRIES = int(os.getenv("MAX_GENERATION_RETRIES", "6"))
GENERATION_BACKOFF_SECONDS = float(os.getenv("GENERATION_BACKOFF_SECONDS", "3"))

ENABLE_NEO4J = os.getenv("ENABLE_NEO4J", "false").lower() == "true"
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
