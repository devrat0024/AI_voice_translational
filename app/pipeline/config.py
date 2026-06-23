"""
app/pipeline/config.py — ETL Pipeline Directory & Format Configuration
"""
from pathlib import Path

# ── Unified data directory (AI_scribe/data/) ─────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # → AI_scribe/
DATA_DIR = BASE_DIR / "data"

# Raw ingestion directories
RAW_DIR = DATA_DIR / "raw"
RAW_AUDIO_DIR = RAW_DIR / "audio"
RAW_PATIENTS_DIR = RAW_DIR / "patients"
RAW_DOCUMENTS_DIR = RAW_DIR / "documents"

# Processed staging directories
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_AUDIO_DIR = PROCESSED_DIR / "audio"
PROCESSED_DOCUMENTS_DIR = PROCESSED_DIR / "documents"

# Archive and failed directories
ARCHIVE_DIR = DATA_DIR / "archive"
FAILED_DIR = DATA_DIR / "failed"

# ETL SQLite database (separate from API's DB to avoid table conflicts)
DATABASE_DIR = DATA_DIR / "database"
DB_PATH = DATABASE_DIR / "scribe_etl.db"

# Audio preprocessing parameters
TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1       # Mono
TARGET_FORMAT = "wav"

# Supported file formats
SUPPORTED_AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac"}
SUPPORTED_PATIENT_EXTENSIONS = {".csv"}
SUPPORTED_DOCUMENT_EXTENSIONS = {".txt", ".md", ".json", ".pdf"}


def init_directories():
    """Creates all required ETL pipeline directories."""
    dirs = [
        RAW_AUDIO_DIR,
        RAW_PATIENTS_DIR,
        RAW_DOCUMENTS_DIR,
        PROCESSED_AUDIO_DIR,
        PROCESSED_DOCUMENTS_DIR,
        ARCHIVE_DIR,
        FAILED_DIR,
        DATABASE_DIR,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
