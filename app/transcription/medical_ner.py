"""
app/transcription/medical_ner.py — Medical Named Entity Recognition
Supports SciSpacy, Hugging Face BioBERT, and rule-based fallback.
"""
import re
import logging
from app.config import SCISPACY_MODEL, NER_TRANSFORMERS_MODEL

logger = logging.getLogger(__name__)

# ── Keyword dictionaries for rule-based fallback ─────────────────────────────
SYMPTOM_KEYWORDS = [
    "fever", "cough", "congestion", "headache", "pain", "nausea", "vomiting",
    "fatigue", "chills", "sore throat", "shortness of breath", "dyspnea", "rash",
    "diarrhea", "dizziness", "weakness",
]

MEDICINE_KEYWORDS = [
    "paracetamol", "acetaminophen", "ibuprofen", "aspirin", "amoxicillin",
    "penicillin", "albuterol", "lipitor", "metformin", "lisinopril", "metoprolol",
    "advil", "tylenol", "antibiotics", "cough syrup",
]


def expand_to_word_boundary(text: str, start: int, end: int) -> tuple:
    """Expands character indices outward to capture full alphanumeric words."""
    while start > 0 and text[start - 1].isalnum():
        start -= 1
    while end < len(text) and text[end].isalnum():
        end += 1
    return start, end


class MedicalEntityExtractor:
    def __init__(self, mode: str = "auto"):
        """Initializes the extractor.

        Mode: 'scispacy' | 'transformer' | 'rules' | 'auto'
        """
        self.mode = mode
        self.nlp_spacy = None
        self.hf_ner_pipeline = None

    def _init_scispacy(self) -> bool:
        """Tries to initialize the SciSpacy NLP model."""
        try:
            import spacy  # type: ignore
            self.nlp_spacy = spacy.load(SCISPACY_MODEL)
            logger.info("SciSpacy model loaded successfully.")
            return True
        except Exception as e:
            logger.warning(f"SciSpacy model '{SCISPACY_MODEL}' unavailable: {e}.")
            return False

    def _init_transformer(self) -> bool:
        """Tries to initialize the Hugging Face clinical NER pipeline."""
        try:
            from transformers import pipeline  # type: ignore
            logger.info(f"Loading Hugging Face NER model '{NER_TRANSFORMERS_MODEL}'...")
            self.hf_ner_pipeline = pipeline(
                "ner", model=NER_TRANSFORMERS_MODEL, aggregation_strategy="simple"
            )
            logger.info("Transformers NER pipeline loaded successfully.")
            return True
        except Exception as e:
            logger.warning(f"Transformers NER pipeline unavailable: {e}.")
            return False

    def extract_via_rules(self, text: str) -> dict:
        """Extracts symptoms and medicines using keyword/regex matching."""
        text_lower = text.lower()
        extracted_symptoms = [
            s.title()
            for s in SYMPTOM_KEYWORDS
            if re.search(r"\b" + re.escape(s) + r"\b", text_lower)
        ]
        extracted_medicines = [
            m.title()
            for m in MEDICINE_KEYWORDS
            if re.search(r"\b" + re.escape(m) + r"\b", text_lower)
        ]
        return {
            "symptoms": list(set(extracted_symptoms)),
            "medicines": list(set(extracted_medicines)),
        }

    def extract_entities(self, text: str) -> dict:
        """Runs entity extraction.

        Returns {'symptom': str, 'medicine': str} — the primary extracted values.
        """
        symptoms, medicines = [], []

        # 1. SciSpacy
        if self.mode in {"scispacy", "auto"}:
            if self.nlp_spacy or self._init_scispacy():
                try:
                    doc = self.nlp_spacy(text)
                    for ent in doc.ents:
                        ent_text = ent.text.strip().title()
                        if any(s.lower() in ent_text.lower() for s in SYMPTOM_KEYWORDS):
                            symptoms.append(ent_text)
                        elif any(m.lower() in ent_text.lower() for m in MEDICINE_KEYWORDS):
                            medicines.append(ent_text)
                except Exception as e:
                    logger.error(f"SciSpacy extraction failed: {e}")

        # 2. Hugging Face transformer NER
        if not symptoms and not medicines and self.mode in {"transformer", "auto"}:
            if self.hf_ner_pipeline or self._init_transformer():
                try:
                    entities = self.hf_ner_pipeline(text)
                    for ent in entities:
                        start = ent.get("start")
                        end = ent.get("end")
                        if start is not None and end is not None:
                            s, e = expand_to_word_boundary(text, start, end)
                            word = text[s:e].strip()
                        else:
                            word = ent["word"].replace("##", "")
                        ent_text = word.strip().title()
                        if len(ent_text) <= 1:
                            continue
                        ent_group = ent.get("entity_group", "").upper()
                        if any(k in ent_group for k in ("SYMPTOM", "SIGN", "DISEASE", "DISORDER")):
                            symptoms.append(ent_text)
                        elif any(k in ent_group for k in ("DRUG", "MEDICINE", "MEDICATION", "CHEMICAL")):
                            medicines.append(ent_text)
                except Exception as e:
                    logger.error(f"Transformer NER extraction failed: {e}")

        # 3. Rule-based fallback (always runs to augment)
        rule_results = self.extract_via_rules(text)
        symptoms.extend(rule_results["symptoms"])
        medicines.extend(rule_results["medicines"])

        symptoms = list(set(symptoms))
        medicines = list(set(medicines))

        return {
            "symptom": symptoms[0] if symptoms else "None",
            "medicine": medicines[0] if medicines else "None",
        }
