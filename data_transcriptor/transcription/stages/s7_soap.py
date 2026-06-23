"""
Stage 7 — SOAP Note Generation
"""
from __future__ import annotations

import re
from typing import Optional

from data_transcriptor.transcription.schemas import SOAPNote, StageStatus
from data_transcriptor.transcription.base_stage import PipelineContext, PipelineStage

_SECTION_PATTERNS = {
    "subjective":  re.compile(r"###?\s*Subjective\s*\n(.*?)(?=###?\s*Objective|\Z)", re.S | re.I),
    "objective":   re.compile(r"###?\s*Objective\s*\n(.*?)(?=###?\s*Assessment|\Z)", re.S | re.I),
    "assessment":  re.compile(r"###?\s*Assessment\s*\n(.*?)(?=###?\s*Plan|\Z)", re.S | re.I),
    "plan":        re.compile(r"###?\s*Plan\s*\n(.*?)(?=###?|\Z)", re.S | re.I),
}


def _parse_soap(raw: str) -> tuple[str, str, str, str]:
    results = {}
    for key, pattern in _SECTION_PATTERNS.items():
        match = pattern.search(raw)
        results[key] = match.group(1).strip() if match else raw.strip()
    return (
        results.get("subjective", ""),
        results.get("objective", ""),
        results.get("assessment", ""),
        results.get("plan", ""),
    )


class SOAPStage(PipelineStage):
    name = "soap_generation"
    description = "Generate structured SOAP clinical note via Groq LLM"
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
        raw_soap = llm.generate_soap_note(text)

        subjective, objective, assessment, plan = _parse_soap(raw_soap)

        soap = SOAPNote(
            subjective=subjective,
            objective=objective,
            assessment=assessment,
            plan=plan,
            raw=raw_soap,
        )
        context["soap_note"] = soap

        status = StageStatus.SUCCESS if llm.use_api else StageStatus.SIMULATED
        return status, f"{'Groq API' if llm.use_api else 'simulation'}"
