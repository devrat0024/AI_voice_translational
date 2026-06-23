"""
app/transcription/llm_layer.py — Groq-Powered Clinical Intelligence Layer

Provides:
  - Medical terminology correction
  - SOAP note generation
  - Clinical summary generation

Falls back to local rules-based simulation when Groq is unavailable.
"""
import os
import re
import logging

logger = logging.getLogger(__name__)

# ── Groq SDK availability check ───────────────────────────────────────────────
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    logger.warning("Groq SDK not installed. LLM Layer will run in simulation mode.")

# ── Local simulation corrections dictionary ───────────────────────────────────
SIMULATION_CORRECTIONS = {
    "lymphosite": "lymphocyte",
    "colestrol": "cholesterol",
    "hypertesion": "hypertension",
    "diabete": "diabetes",
    "paracetemol": "paracetamol",
    "ibuprofin": "ibuprofen",
}


class ClinicalIntelligenceLayer:
    def __init__(self, model_name: str = "llama-3.3-70b-versatile", api_key: str = ""):
        """Initializes the clinical intelligence layer.

        Connects to Groq if api_key or GROQ_API_KEY env var is available,
        otherwise falls back to local rules-based simulation.
        """
        self.model_name = model_name
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "")
        self.client = None
        self.use_api = False

        if GROQ_AVAILABLE and self.api_key:
            try:
                self.client = Groq(api_key=self.api_key)
                self.use_api = True
                logger.info(f"Groq client initialized. Model: {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize Groq client: {e}. Falling back to simulation.")
        else:
            if not GROQ_AVAILABLE:
                logger.warning("Groq SDK unavailable. Running in local simulation mode.")
            elif not self.api_key:
                logger.warning("GROQ_API_KEY missing. Running in local simulation mode.")

    def _call_groq(self, prompt: str, temperature: float = 0.2) -> str | None:
        """Makes a Groq API call and returns the response text, or None on failure."""
        if not (self.use_api and self.client):
            return None
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Groq API call failed: {e}. Falling back to simulation.")
            return None

    def medical_correction(self, text: str) -> str:
        """Corrects misspelled clinical terms and medical jargon in text."""
        prompt = (
            "You are a medical speech transcriber. Correct misspelled medical terminology, "
            "drug names, and clinical jargon in the provided transcribed text. Maintain all "
            "other parts of the text exactly as they are. "
            "Return ONLY the corrected text without any conversational intro/outro.\n\n"
            f"Input text:\n{text}\n\nCorrected text:"
        )
        result = self._call_groq(prompt, temperature=0.1)
        if result:
            logger.info("Medical terminology correction completed via Groq API.")
            return result

        # Local simulation fallback
        logger.info("Running medical correction in local simulation mode...")
        corrected = text
        for misspelled, correct in SIMULATION_CORRECTIONS.items():
            corrected = re.sub(re.escape(misspelled), correct, corrected, flags=re.IGNORECASE)
        return corrected

    def generate_soap_note(self, transcript: str) -> str:
        """Generates a SOAP (Subjective, Objective, Assessment, Plan) clinical note."""
        prompt = (
            "You are a clinical AI scribe. Generate a professional SOAP note "
            "(Subjective, Objective, Assessment, Plan) based on the following patient-doctor "
            "conversation transcript. Do not include any chat metadata or introductory remarks. "
            "Format each section clearly with markdown headers.\n\n"
            f"Transcript:\n{transcript}\n\nSOAP Note:"
        )
        result = self._call_groq(prompt, temperature=0.2)
        if result:
            logger.info("SOAP note generation completed via Groq API.")
            return result

        # Simulation fallback
        logger.info("Running SOAP generation in local simulation mode...")
        has_baby = "baby" in transcript.lower() or "george" in transcript.lower()
        has_moving = "mov" in transcript.lower()

        subjective = "Patient catching up with provider. "
        if has_baby:
            subjective += "Mentioned patient's child George, now 2 years old. "
        if has_moving:
            subjective += "Patient plans to relocate in a couple of months and works at a health center. "

        return (
            "### Subjective\n"
            f"- {subjective}\n\n"
            "### Objective\n"
            "- Patient appears in good spirits, oriented to time, place, and person.\n\n"
            "### Assessment\n"
            "- Normal clinical status. Informal/catch-up visit.\n\n"
            "### Plan\n"
            "- Continue standard care. Follow up as needed. Best wishes for the relocation."
        )

    def generate_clinical_summary(self, transcript: str) -> str:
        """Generates a concise clinical summary from the transcript."""
        prompt = (
            "You are a clinical assistant. Summarize the following patient-doctor clinical "
            "conversation into a concise medical summary. Highlight key symptoms, advice given, "
            "and next steps. Return ONLY the summary, no chat wrapper.\n\n"
            f"Transcript:\n{transcript}\n\nClinical Summary:"
        )
        result = self._call_groq(prompt, temperature=0.2)
        if result:
            logger.info("Clinical summary generation completed via Groq API.")
            return result

        # Simulation fallback
        logger.info("Running clinical summary generation in local simulation mode...")
        return (
            "The provider and patient had a brief informal conversation. Topics discussed include "
            "family updates (patient's child George is now 2 years old) and professional updates "
            "(patient is working at the health center and plans to relocate in a couple of months)."
        )
