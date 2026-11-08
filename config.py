"""Project-wide paths and settings loaded from the environment."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
RESUME_PDF_PATH = DATA_DIR / "resume.pdf"
QA_BANK_PATH = DATA_DIR / "qa_bank.json"
FAISS_INDEX_DIR = DATA_DIR / "faiss_index"
FAISS_INDEX_PATH = FAISS_INDEX_DIR / "index.faiss"
FAISS_METADATA_PATH = FAISS_INDEX_DIR / "metadata.json"
CHUNK_PREVIEW_PATH = FAISS_INDEX_DIR / "chunk_preview.html"

COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
COHERE_EMBEDDING_MODEL = os.getenv("COHERE_EMBEDDING_MODEL", "embed-english-v3.0")
COHERE_CHAT_MODEL = os.getenv("COHERE_CHAT_MODEL", "command-r-plus")

# embed-english-v3.0 float embeddings are 1024-d
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))
