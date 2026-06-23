import sys
import os
from app.config import TRANSCRIPTOR_DIR

# Ensure transcription package imports work
if str(TRANSCRIPTOR_DIR) not in sys.path:
    sys.path.append(str(TRANSCRIPTOR_DIR))

from transcription.llm_layer import ClinicalIntelligenceLayer

# Instantiate the intelligence layer (automatically picks up environment credentials)
groq_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
intel_layer = ClinicalIntelligenceLayer(model_name=groq_model)

def run_medical_correction(text: str) -> str:
    """Performs clinical spelling correction on raw text."""
    return intel_layer.medical_correction(text)

def run_soap_generation(text: str) -> str:
    """Generates a structured SOAP clinical note from text."""
    return intel_layer.generate_soap_note(text)

def run_clinical_summary(text: str) -> str:
    """Generates a clinical summary from text."""
    return intel_layer.generate_clinical_summary(text)
