"""
app/core/runner.py — Clinical AI Pipeline Orchestrator

Runs all 8 stages sequentially, collects typed results, and assembles
the final PipelineResult. Saves JSON output automatically.

Usage:
    from app.core.runner import ClinicalPipeline
    from app.core.schemas import PipelineConfig

    config = PipelineConfig(whisper_model="base", groq_api_key="...")
    pipeline = ClinicalPipeline(config)
    result = pipeline.run("path/to/audio.mp3")

    print(result.model_dump_json(indent=2))
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import List, Optional

from app.core.schemas import (
    ClinicalIntelligenceResult,
    PipelineConfig,
    PipelineMetadata,
    PipelineResult,
    StageStatus,
    TranscriptionResult,
)
from app.core.stages.base import PipelineContext, PipelineStage
from app.core.stages.s1_load import AudioLoadStage
from app.core.stages.s2_transcribe import TranscriptionStage
from app.core.stages.s3_diarize import DiarizationStage
from app.core.stages.s4_align import AlignmentStage
from app.core.stages.s5_correct import MedicalCorrectionStage
from app.core.stages.s6_ner import NERStage
from app.core.stages.s7_soap import SOAPStage
from app.core.stages.s8_summary import ClinicalSummaryStage

logger = logging.getLogger(__name__)

# The ordered list of pipeline stages
_STAGES: List[PipelineStage] = [
    AudioLoadStage(),
    TranscriptionStage(),
    DiarizationStage(),
    AlignmentStage(),
    MedicalCorrectionStage(),
    NERStage(),
    SOAPStage(),
    ClinicalSummaryStage(),
]


class ClinicalPipeline:
    """
    Structured Clinical AI Pipeline.

    Runs 8 sequential stages on a clinical audio file and produces
    a fully typed PipelineResult with rich structured JSON output.

    Stages:
      1. AudioLoad        — Validate file, extract metadata
      2. Transcription    — Whisper ASR
      3. Diarization      — Speaker identification (pyannote / simulation)
      4. Alignment        — Timestamp-overlap speaker-segment mapping
      5. MedCorrection    — Groq LLM medical spelling correction
      6. NER              — Medical named entity extraction
      7. SOAP             — Groq LLM SOAP note generation
      8. ClinicalSummary  — Groq LLM encounter summary
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.stages = _STAGES

    def run(self, audio_path: str | Path) -> PipelineResult:
        """
        Execute the full pipeline on the given audio file.

        Args:
            audio_path: Path to the clinical audio file.

        Returns:
            PipelineResult — fully populated structured result.
        """
        audio_path = Path(audio_path)
        pipeline_start = time.perf_counter()

        logger.info("=" * 60)
        logger.info(f"Clinical AI Pipeline starting for: {audio_path.name}")
        logger.info(f"Config: whisper={self.config.whisper_model}, "
                    f"ner={self.config.ner_mode}, "
                    f"groq_model={self.config.groq_model}")
        logger.info("=" * 60)

        # Shared mutable context — stages read/write their outputs here
        context: PipelineContext = {
            "config": self.config,
            "audio_path": audio_path,
        }

        stage_results = []
        pipeline_failed = False

        for stage in self.stages:
            if pipeline_failed and not stage.optional:
                from app.core.schemas import StageResult
                stage_results.append(
                    StageResult(
                        name=stage.name,
                        status=StageStatus.SKIPPED,
                        duration_ms=0.0,
                        message="Skipped due to earlier critical failure",
                    )
                )
                continue

            result = stage.run(context)
            stage_results.append(result)

            if result.status == StageStatus.FAILED and not stage.optional:
                if not self.config.continue_on_error:
                    pipeline_failed = True
                    logger.error(
                        f"Critical stage '{stage.name}' failed. "
                        f"Aborting remaining stages."
                    )

        # ── Assemble final PipelineResult ─────────────────────────────────────
        total_elapsed = time.perf_counter() - pipeline_start
        succeeded = sum(1 for r in stage_results if r.status in (StageStatus.SUCCESS, StageStatus.SIMULATED))
        failed = sum(1 for r in stage_results if r.status == StageStatus.FAILED)

        metadata = PipelineMetadata(
            audio_file=audio_path.name,
            total_duration_seconds=round(total_elapsed, 3),
            stages_run=len(stage_results),
            stages_succeeded=succeeded,
            stages_failed=failed,
            llm_mode=context.get("llm_mode", "unknown"),
        )

        # TranscriptionResult
        transcription_result = None
        if "raw_text" in context:
            transcription_result = TranscriptionResult(
                raw_text=context.get("raw_text", ""),
                corrected_text=context.get("corrected_text") or context.get("raw_text", ""),
                segment_count=len(context.get("segments", [])),
                dialogue=context.get("dialogue", []),
            )

        # ClinicalIntelligenceResult
        clinical_result = None
        if "soap_note" in context and "clinical_summary" in context:
            clinical_result = ClinicalIntelligenceResult(
                soap_note=context["soap_note"],
                clinical_summary=context["clinical_summary"],
            )

        result = PipelineResult(
            metadata=metadata,
            stages=stage_results,
            audio=context.get("audio_metadata"),
            transcription=transcription_result,
            medical_entities=context.get("medical_entities"),
            clinical_intelligence=clinical_result,
        )

        self._log_summary(result)
        self._save_output(result, audio_path)

        return result

    def _log_summary(self, result: PipelineResult) -> None:
        """Logs a formatted pipeline summary to the console."""
        logger.info("")
        logger.info("╔══════════════════════════════════════════════════════╗")
        logger.info("║          CLINICAL AI PIPELINE COMPLETE               ║")
        logger.info("╠══════════════════════════════════════════════════════╣")
        logger.info(f"║  File:      {result.metadata.audio_file:<42}║")
        logger.info(f"║  Duration:  {result.metadata.total_duration_seconds:.1f}s total{' ' * 37}║")
        logger.info(f"║  LLM Mode:  {result.metadata.llm_mode:<42}║")
        logger.info(f"║  Stages:    {result.metadata.stages_succeeded} succeeded, "
                    f"{result.metadata.stages_failed} failed{' ' * 30}║")
        logger.info("╠══════════════════════════════════════════════════════╣")
        for stage in result.stages:
            icon = {"success": "✓", "simulated": "~", "failed": "✗", "skipped": "○"}.get(
                stage.status.value, "?"
            )
            logger.info(
                f"║  {icon} {stage.name:<22} {stage.status.value:<10} {stage.duration_ms:>7.0f}ms  ║"
            )
        logger.info("╚══════════════════════════════════════════════════════╝")

    def _save_output(self, result: PipelineResult, audio_path: Path) -> None:
        """Saves the PipelineResult to a structured JSON file."""
        from app.config import OUTPUT_DIR, init_directories
        init_directories()

        if self.config.output_path:
            output_file = Path(self.config.output_path)
        else:
            output_file = OUTPUT_DIR / f"{audio_path.stem}_pipeline_result.json"

        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(result.to_json())

        logger.info(f"Structured output saved: {output_file.resolve()}")
