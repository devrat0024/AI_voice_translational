"""
app/config.py — Unified Configuration
Merged from:
  - backend/app/config.py
  - data_transcriptor/transcription/config.py
"""
import os
import sys
from pathlib import Path

# ── Attempt to load static_ffmpeg so pydub/whisper can find ffmpeg ──────────
try:
    import static_ffmpeg
    static_ffmpeg.add_paths()
except ImportError:
    pass  # static_ffmpeg is optional at import time; it will be needed at runtime

# ── Directory Layout ─────────────────────────────────────────────────────────
# AI_scribe/ (root)
BASE_DIR = Path(__file__).resolve().parent.parent

# Unified data directory
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
OUTPUT_DIR = DATA_DIR / "output"          # Transcription JSON/TXT outputs
UPLOAD_DIR = BASE_DIR / "uploads"         # FastAPI audio uploads


def init_directories():
    """Creates all required runtime directories."""
    dirs = [
        RAW_DIR,
        OUTPUT_DIR,
        UPLOAD_DIR,
        BASE_DIR / "data" / "database",   # For API's clinical_scribe.db
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


# ── .env Loader ──────────────────────────────────────────────────────────────
def load_env_file():
    """Lightweight .env file loader — sets os.environ from KEY=VALUE lines."""
    search_paths = [
        Path(".env"),
        BASE_DIR / ".env",
    ]
    for path in search_paths:
        if path.is_file():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        if "=" in line:
                            key, val = line.split("=", 1)
                            key = key.strip()
                            val = val.strip().strip("'\"")
                            os.environ[key] = val
                            os.environ[key.upper()] = val
                break
            except Exception:
                pass


# Load .env at import time
load_env_file()

# ── Authentication Settings (FastAPI JWT) ────────────────────────────────────
SECRET_KEY = os.getenv("SECRET_KEY", "super_secret_clinical_key_123!")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))

# ── Database ─────────────────────────────────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'data' / 'database' / 'clinical_scribe.db'}")

# ── Whisper / Diarization ────────────────────────────────────────────────────
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL_NAME", "tiny")
HF_TOKEN = os.getenv("HF_TOKEN", "") or os.getenv("HF_TOKEN", "")

# ── Medical NER Models ───────────────────────────────────────────────────────
SCISPACY_MODEL = "en_core_sci_sm"
NER_TRANSFORMERS_MODEL = "d4data/biomedical-ner-all"

# ── Groq LLM ─────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
