"""Application configuration for LawLens.

All paths are resolved from this file's project directory so the app behaves
the same way regardless of the terminal's current directory.
"""

from pathlib import Path
import os


PROJECT_DIR = Path(__file__).resolve().parent
LEGAL_DOCS_DIR = PROJECT_DIR / "legal_docs"
DATA_DIR = PROJECT_DIR / "data"
INCIDENTS_FILE = DATA_DIR / "incidents.json"
STATIC_DIR = PROJECT_DIR / "static"

OLLAMA_HOST = os.getenv("LAW_LENS_OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("LAW_LENS_OLLAMA_MODEL", "qwen2.5:3b")
CHUNK_SIZE = int(os.getenv("LAW_LENS_CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("LAW_LENS_CHUNK_OVERLAP", "200"))
TOP_K = int(os.getenv("LAW_LENS_TOP_K", "5"))
RETRIEVAL_THRESHOLD = float(os.getenv("LAW_LENS_RETRIEVAL_THRESHOLD", "0.08"))
MAX_CONTEXT_CHARS = int(os.getenv("LAW_LENS_MAX_CONTEXT_CHARS", "12000"))
EMERGENCY_GUIDANCE_SOURCE = "emergency_financial_fraud_guidance.txt"
FINANCIAL_FIRST_STEPS_SOURCE = "financial_fraud_first_steps.txt"
NCRP_SOURCE_URL = "https://www.cybercrime.gov.in/"
NCRP_VERIFIED_DATE = "2026-10-11"
