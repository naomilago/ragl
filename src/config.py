'''
  Central configuration for the local RAG pipeline.
  Change models/params here instead of hunting through every module.
'''

from pathlib import Path

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
CHROMA_DIR = BASE_DIR / 'chroma_db'

# --- Ollama models (via langchain-ollama) ---
LLM_MODEL = 'qwen3.5:9b'
EMBED_MODEL = 'nomic-embed-text'

# --- Chunking ---
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# --- Retrieval ---
TOP_K = 4

# --- ChromaDB (via langchain-chroma) ---
COLLECTION_NAME = 'documents'