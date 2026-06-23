"""
Stage 5 — Medical Terminology Correction (Groq LLM / Rules Fallback)

Corrects misspelled medical terms, drug names, and clinical jargon.
Writes to context: 'corrected_text', 'llm_mode'
"""
from __future__ import annotations

from typing import Optional

from app.core.schemas import StageStatus
from app.core.stages.base import PipelineContext, PipelineStage


class MedicalCorrectionStage(PipelineStage):
    name = "med_correction"
    description = "Correct medical terminology via Groq LLM or rules-based fallback"
    optional = True

    def _execute(self, context: PipelineContext) -> tuple[StageStatus, Optional[str]]:
        from app.transcription.llm_layer import ClinicalIntelligenceLayer

        config = context["config"]
        raw_text: str = context.get("raw_text", "")

        llm = ClinicalIntelligenceLayer(
            model_name=config.groq_model,
            api_key=config.groq_api_key,
        )

        corrected = llm.medical_correction(raw_text)
        context["corrected_text"] = corrected
        context["llm_layer"] = llm  # Reuse in subsequent LLM stages

        # Detect whether Groq API was actually used
        used_api = llm.use_api
        context["llm_mode"] = "groq_api" if used_api else "simulation"

        status = StageStatus.SUCCESS if used_api else StageStatus.SIMULATED
        mode_label = "Groq API" if used_api else "rules-based simulation"
        return status, f"Correction via {mode_label}"
