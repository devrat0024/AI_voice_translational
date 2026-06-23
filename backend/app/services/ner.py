import sys
from app.config import TRANSCRIPTOR_DIR

# Ensure transcription package imports work
if str(TRANSCRIPTOR_DIR) not in sys.path:
    sys.path.append(str(TRANSCRIPTOR_DIR))

from transcription.medical_ner import MedicalEntityExtractor

# Instantiate once at startup to optimize memory loading
extractor = MedicalEntityExtractor(mode="auto")

def run_ner_extraction(text: str) -> dict:
    """Extracts clinical entities (symptoms and medicines) from raw text."""
    return extractor.extract_entities(text)
