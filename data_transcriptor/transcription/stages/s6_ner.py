"""
Stage 6 — Medical Named Entity Recognition (NER)
"""
from __future__ import annotations

from typing import Optional

from data_transcriptor.transcription.schemas import MedicalEntities, StageStatus
from data_transcriptor.transcription.base_stage import PipelineContext, PipelineStage


class NERStage(PipelineStage):
    name = "ner_extraction"
    description = "Extract medical entities: symptoms and medicines"
    optional = True

    def _execute(self, context: PipelineContext) -> tuple[StageStatus, Optional[str]]:
        from data_transcriptor.transcription.medical_ner import MedicalEntityExtractor

        config = context["config"]
        text = context.get("corrected_text") or context.get("raw_text", "")

        extractor = MedicalEntityExtractor(mode=config.ner_mode)
        raw_entities = extractor.extract_entities(text)

        symptom_val = raw_entities.get("symptom", "None")
        medicine_val = raw_entities.get("medicine", "None")

        symptoms = [symptom_val] if symptom_val != "None" else []
        medicines = [medicine_val] if medicine_val != "None" else []

        entities = MedicalEntities(
            symptoms=symptoms,
            medicines=medicines,
            primary_symptom=symptom_val,
            primary_medicine=medicine_val,
        )
        context["medical_entities"] = entities

        found = []
        if symptoms:
            found.append(f"symptoms: {symptoms}")
        if medicines:
            found.append(f"medicines: {medicines}")
        msg = ", ".join(found) if found else "No entities detected"
        return StageStatus.SUCCESS, msg
