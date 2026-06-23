import re
import logging
from .config import SCISPACY_MODEL, NER_TRANSFORMERS_MODEL


logger = logging.getLogger(__name__)

# Predefined medical terminology dictionary for robust rule-based parsing & fallback
SYMPTOM_KEYWORDS = [
    "fever", "cough", "congestion", "headache", "pain", "nausea", "vomiting",
    "fatigue", "chills", "sore throat", "shortness of breath", "dyspnea", "rash",
    "diarrhea", "dizziness", "weakness"
]

MEDICINE_KEYWORDS = [
    "paracetamol", "acetaminophen", "ibuprofen", "aspirin", "amoxicillin",
    "penicillin", "albuterol", "lipitor", "metformin", "lisinopril", "metoprolol",
    "advil", "tylenol", "antibiotics", "cough syrup"
]

def expand_to_word_boundary(text: str, start: int, end: int) -> tuple:
    """Expands character indices outward to capture full alnum words."""
    while start > 0 and text[start - 1].isalnum():
        start -= 1
    while end < len(text) and text[end].isalnum():
        end += 1
    return start, end

class MedicalEntityExtractor:
    def __init__(self, mode: str = "auto"):
        """Initializes the entity extractor.

        Mode can be 'scispacy', 'transformer', 'rules', or 'auto'.
        """
        self.mode = mode
        self.nlp_spacy = None
        self.hf_ner_pipeline = None

    def _init_scispacy(self) -> bool:
        """Tries to initialize spaCy and SciSpacy model."""
        try:
            import spacy # type: ignore # pylint: disable=import-error
            # Suppress warning, load scispacy if available
            self.nlp_spacy = spacy.load(SCISPACY_MODEL)
            logger.info("SciSpacy model loaded successfully.")
            return True
        except Exception as e:
            logger.warning(f"SciSpacy model '{SCISPACY_MODEL}' not available: {e}.")
            return False

    def _init_transformer(self) -> bool:
        """Tries to initialize Hugging Face transformers pipeline."""
        try:
            from transformers import pipeline
            logger.info(f"Loading Hugging Face clinical NER model '{NER_TRANSFORMERS_MODEL}'...")
            self.hf_ner_pipeline = pipeline("ner", model=NER_TRANSFORMERS_MODEL, aggregation_strategy="simple")
            logger.info("Transformers clinical NER pipeline loaded successfully.")
            return True
        except Exception as e:
            logger.warning(f"Transformers clinical NER pipeline not available: {e}.")
            return False

    def extract_via_rules(self, text: str) -> dict:
        """Extracts symptoms and medicines using regex and keyword dictionary matching."""
        extracted_symptoms = []
        extracted_medicines = []

        text_lower = text.lower()

        # Keyword matching
        for symptom in SYMPTOM_KEYWORDS:
            # Match word boundary
            if re.search(r'\b' + re.escape(symptom) + r'\b', text_lower):
                extracted_symptoms.append(symptom.title())

        for medicine in MEDICINE_KEYWORDS:
            if re.search(r'\b' + re.escape(medicine) + r'\b', text_lower):
                extracted_medicines.append(medicine.title())

        return {
            "symptoms": list(set(extracted_symptoms)),
            "medicines": list(set(extracted_medicines))
        }

    def extract_entities(self, text: str) -> dict:
        """Performs entity extraction based on the configured mode."""
        symptoms = []
        medicines = []

        # 1. Try SciSpacy if configured or auto
        if self.mode in {"scispacy", "auto"}:
            if self.nlp_spacy or self._init_scispacy():
                try:
                    doc = self.nlp_spacy(text)
                    for ent in doc.ents:
                        # SciSpacy extracts clinical entities - we'll filter or categorize them
                        # Based on simple keyword mapping for labels
                        ent_text = ent.text.strip().title()
                        # Simple heuristics for demonstration
                        if any(s.lower() in ent_text.lower() for s in SYMPTOM_KEYWORDS):
                            symptoms.append(ent_text)
                        elif any(m.lower() in ent_text.lower() for m in MEDICINE_KEYWORDS):
                            medicines.append(ent_text)
                except Exception as e:
                    logger.error(f"SciSpacy extraction failed: {e}")

        # 2. Try Hugging Face clinical NER pipeline
        if not symptoms and not medicines and self.mode in {"transformer", "auto"}:
            if self.hf_ner_pipeline or self._init_transformer():
                try:
                    entities = self.hf_ner_pipeline(text)
                    for ent in entities:
                        start = ent.get('start')
                        end = ent.get('end')
                        if start is not None and end is not None:
                            start_exp, end_exp = expand_to_word_boundary(text, start, end)
                            word = text[start_exp:end_exp].strip()
                        else:
                            word = ent['word'].replace("##", "")
                        
                        ent_text = word.strip().title()
                        ent_group = ent.get('entity_group', '').upper()
                        
                        if len(ent_text) <= 1:
                            continue
                            
                        # Map standard clinical labels (e.g. SIGN_SYMPTOM, DOSAGE, DRUG, MEDICINE, MEDICATION)
                        if 'SYMPTOM' in ent_group or 'SIGN' in ent_group or 'DISEASE' in ent_group or 'DISORDER' in ent_group:
                            symptoms.append(ent_text)
                        elif 'DRUG' in ent_group or 'MEDICINE' in ent_group or 'MEDICATION' in ent_group or 'CHEMICAL' in ent_group:
                            medicines.append(ent_text)
                except Exception as e:
                    logger.error(f"Transformer NER extraction failed: {e}")

        # 3. Fallback to dictionary rules
        rule_results = self.extract_via_rules(text)
        symptoms.extend(rule_results["symptoms"])
        medicines.extend(rule_results["medicines"])

        # De-duplicate
        symptoms = list(set(symptoms))
        medicines = list(set(medicines))

        # Formatting output as requested:
        # { "symptom": "Fever", "medicine": "Paracetamol" }
        return {
            "symptom": symptoms[0] if symptoms else "None",
            "medicine": medicines[0] if medicines else "None"
        }
