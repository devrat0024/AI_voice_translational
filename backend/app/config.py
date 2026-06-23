"""
backend/app/config.py — Unified Configuration Manager
"""
import os
from pathlib import Path

# ── Attempt to load static_ffmpeg so pydub/whisper can find ffmpeg ──────────
try:
    import static_ffmpeg
    static_ffmpeg.add_paths()
except ImportError:
    pass

# BASE_DIR should resolve to the project root (AI_scribe/)
BASE_DIR = Path(__file__).resolve().parent.parent.parent

import sys
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Unified data directories
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
OUTPUT_DIR = DATA_DIR / "output"
UPLOAD_DIR = BASE_DIR / "uploads"


def init_directories():
    """Creates all required runtime directories."""
    dirs = [
        RAW_DIR,
        OUTPUT_DIR,
        UPLOAD_DIR,
        DATA_DIR / "database",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def load_env_file():
    """Loads environment variables from .env file at the project root."""
    env_path = BASE_DIR / ".env"
    if env_path.is_file():
        try:
            with open(env_path, "r", encoding="utf-8") as f:
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
        except Exception:
            pass


# Initialize config environment
load_env_file()
init_directories()

# ── Authentication Settings (FastAPI JWT) ────────────────────────────────────
SECRET_KEY = os.getenv("SECRET_KEY", "super_secret_clinical_key_123!")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))

# ── Database ─────────────────────────────────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DATA_DIR / 'database' / 'clinical_scribe.db'}")

# ── Whisper / Diarization ────────────────────────────────────────────────────
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL_NAME", "tiny")
HF_TOKEN = os.getenv("HF_TOKEN", "")

# ── Medical NER Models ───────────────────────────────────────────────────────
SCISPACY_MODEL = "en_core_sci_sm"
NER_TRANSFORMERS_MODEL = "d4data/biomedical-ner-all"

# ── Groq LLM ─────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
