import os
from pathlib import Path
import static_ffmpeg

static_ffmpeg.add_paths()

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
OUTPUT_DIR = DATA_DIR / "output"

def load_env_file():
    """Lightweight .env file loader that sets environment variables."""
    search_paths = [
        Path(".env"),
        Path("../.env"),
        BASE_DIR / ".env",
        BASE_DIR.parent / ".env"
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
                            # Set both original and uppercase variants
                            os.environ[key] = val
                            os.environ[key.upper()] = val
                break
            except Exception:
                pass

# Load environment variables from .env
load_env_file()

# Model Configurations
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL_NAME", "tiny")
HF_TOKEN = os.getenv("HF_TOKEN", "") # Required for pyannote.audio pretrained pipeline

# Medical NER Fallback Configuration
# We can use scispacy, transformers BioBERT/ClinicalBERT, or rule-based models
SCISPACY_MODEL = "en_core_sci_sm"
NER_TRANSFORMERS_MODEL = "d4data/biomedical-ner-all" # Public Biomedical NER model

def init_directories():
    """Initializes raw and output directories."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
