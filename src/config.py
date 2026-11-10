"""
config.py
Configuration loader and management for the PDF-based RAG application.
Loads settings from environment variables or a .env file with sensible defaults.
"""

import os
from pathlib import Path
from dataclasses import dataclass

# Try loading python-dotenv if installed
try:
    from dotenv import load_dotenv
    # Load .env from project root
    env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(dotenv_path=env_path)
except ImportError:
    pass


@dataclass
class AppConfig:
    """Central configuration for the RAG pipeline."""

    # Project directories
    project_root: Path = Path(__file__).resolve().parent.parent
    documents_dir: Path = project_root / os.getenv("DOCUMENTS_DIR", "./documents")
    chroma_db_dir: Path = project_root / os.getenv("CHROMA_DB_DIR", "./chroma_db")

    # ChromaDB configuration
    collection_name: str = os.getenv("CHROMA_COLLECTION_NAME", "rag_documents")

    # Embedding settings
    embedding_model_name: str = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

    # Chunking settings
    # In natural language, ~4 characters roughly equal 1 token.
    # 500 tokens ≈ 2000 chars, 100 tokens overlap ≈ 400 chars.
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "2000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "400"))

    # Retrieval settings
    top_k: int = int(os.getenv("TOP_K", "6"))
    relevance_threshold: float = float(os.getenv("RELEVANCE_THRESHOLD", "0.80"))

    # LLM Settings (Provider: 'groq' or 'ollama')
    llm_provider: str = os.getenv("LLM_PROVIDER", "groq").lower()

    # Groq (Online Cloud LLM) settings
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
    groq_base_url: str = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")

    # Local LLM (Ollama) settings
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
    ollama_timeout: int = int(os.getenv("OLLAMA_TIMEOUT", "120"))

    # Redis Caching Settings
    redis_enabled: bool = os.getenv("REDIS_ENABLED", "True").lower() == "true"
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    redis_ttl: int = int(os.getenv("REDIS_TTL", "3600"))
    redis_strategy: str = os.getenv("REDIS_STRATEGY", "cache_aside").lower()
    redis_eviction_policy: str = os.getenv("REDIS_EVICTION_POLICY", "allkeys-lru").lower()
    redis_use_fallback: bool = os.getenv("REDIS_USE_FALLBACK", "True").lower() == "true"

    def ensure_directories(self) -> None:
        """Ensure required storage directories exist."""
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_db_dir.mkdir(parents=True, exist_ok=True)


# Singleton instance for convenience
config = AppConfig()

