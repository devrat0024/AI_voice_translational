"""
backend/app/services/ner.py
LEGACY FILE — now delegates to the unified app package.
"""
from app.transcription.medical_ner import MedicalEntityExtractor

# Instantiate once at startup to optimize memory loading
extractor = MedicalEntityExtractor(mode="auto")


def run_ner_extraction(text: str) -> dict:
    """Extracts clinical entities (symptoms and medicines) from raw text."""
    return extractor.extract_entities(text)
