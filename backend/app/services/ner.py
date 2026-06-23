"""
backend/app/services/ner.py
LEGACY FILE — now delegates to the unified transcription package.
"""
from data_transcriptor.transcription.medical_ner import MedicalEntityExtractor

# Instantiate once at startup to optimize memory loading
extractor = MedicalEntityExtractor(mode="auto")


def run_ner_extraction(text: str) -> dict:
    """Extracts clinical entities (symptoms and medicines) from raw text."""
    entities = extractor.extract_entities(text)
    return {
        "symptoms": entities.symptoms,
        "medicines": entities.medicines
    }
