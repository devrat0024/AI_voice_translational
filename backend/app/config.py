import os
import sys
from pathlib import Path

# Ensure data_transcriptor is in Python Path for imports
BASE_DIR = Path(__file__).resolve().parent.parent
TRANSCRIPTOR_DIR = BASE_DIR.parent / "data_transcriptor"
if str(TRANSCRIPTOR_DIR) not in sys.path:
    sys.path.append(str(TRANSCRIPTOR_DIR))

# Use the shared .env loader if available
try:
    from transcription.config import load_env_file
    load_env_file()
except Exception:
    pass

# Authentication Settings
SECRET_KEY = os.getenv("SECRET_KEY", "super_secret_clinical_key_123!")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))

# Database & Storage
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./clinical_scribe.db")
UPLOAD_DIR = BASE_DIR / "uploads"

# Ensure directories exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
