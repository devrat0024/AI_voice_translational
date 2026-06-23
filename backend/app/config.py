"""
backend/app/config.py
LEGACY FILE — now delegates to the unified app package.
All configuration is centrally managed in app/config.py.
"""
from app.config import (
    SECRET_KEY,
    ALGORITHM,
    ACCESS_TOKEN_EXPIRE_MINUTES,
    DATABASE_URL,
    UPLOAD_DIR,
    HF_TOKEN,
    load_env_file,
)

# Ensure .env is loaded and upload dir exists
load_env_file()
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Keep TRANSCRIPTOR_DIR as a stub so any old reference doesn't crash
from pathlib import Path
TRANSCRIPTOR_DIR = Path(__file__).resolve().parent.parent.parent / "data_transcriptor"
