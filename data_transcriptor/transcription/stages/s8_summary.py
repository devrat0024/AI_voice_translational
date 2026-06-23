"""
Stage 8 — Clinical Summary Generation
"""
from __future__ import annotations

from typing import Optional

from data_transcriptor.transcription.schemas import StageStatus
from data_transcriptor.transcription.base_stage import PipelineContext, PipelineStage


class ClinicalSummaryStage(PipelineStage):
    name = "clinical_summary"
    description = "Generate concise clinical encounter summary via Groq LLM"
    optional = True

    def _execute(self, context: PipelineContext) -> tuple[StageStatus, Optional[str]]:
        llm = context.get("llm_layer")
        if llm is None:
            from data_transcriptor.transcription.llm_layer import ClinicalIntelligenceLayer
            config = context["config"]
            llm = ClinicalIntelligenceLayer(
                model_name=config.groq_model,
                api_key=config.groq_api_key,
            )

        text = context.get("corrected_text") or context.get("raw_text", "")
        summary = llm.generate_clinical_summary(text)

        context["clinical_summary"] = summary

        status = StageStatus.SUCCESS if llm.use_api else StageStatus.SIMULATED
        return status, f"{len(summary)} chars via {'Groq API' if llm.use_api else 'simulation'}"
