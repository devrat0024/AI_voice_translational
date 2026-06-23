"""
data_transcriptor/transcription/runner.py — Clinical AI Pipeline Orchestrator
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List, Optional

from data_transcriptor.transcription.schemas import (
    ClinicalIntelligenceResult,
    PipelineConfig,
    PipelineMetadata,
    PipelineResult,
    StageStatus,
    TranscriptionResult,
)
from data_transcriptor.transcription.base_stage import PipelineContext, PipelineStage
from data_transcriptor.transcription.stages.s1_load import AudioLoadStage
from data_transcriptor.transcription.stages.s2_transcribe import TranscriptionStage
from data_transcriptor.transcription.stages.s3_diarize import DiarizationStage
from data_transcriptor.transcription.stages.s4_align import AlignmentStage
from data_transcriptor.transcription.stages.s5_correct import MedicalCorrectionStage
from data_transcriptor.transcription.stages.s6_ner import NERStage
from data_transcriptor.transcription.stages.s7_soap import SOAPStage
from data_transcriptor.transcription.stages.s8_summary import ClinicalSummaryStage

logger = logging.getLogger(__name__)

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
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.stages = _STAGES

    def run(self, audio_path: str | Path) -> PipelineResult:
        audio_path = Path(audio_path)
        pipeline_start = time.perf_counter()

        logger.info("=" * 60)
        logger.info(f"Clinical AI Pipeline starting for: {audio_path.name}")
        logger.info(f"Config: whisper={self.config.whisper_model}, "
                    f"ner={self.config.ner_mode}, "
                    f"groq_model={self.config.groq_model}")
        logger.info("=" * 60)

        context: PipelineContext = {
            "config": self.config,
            "audio_path": audio_path,
        }

        stage_results = []
        pipeline_failed = False

        for stage in self.stages:
            if pipeline_failed and not stage.optional:
                from data_transcriptor.transcription.schemas import StageResult
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

        transcription_result = None
        if "raw_text" in context:
            transcription_result = TranscriptionResult(
                raw_text=context.get("raw_text", ""),
                corrected_text=context.get("corrected_text") or context.get("raw_text", ""),
                segment_count=len(context.get("segments", [])),
                dialogue=context.get("dialogue", []),
            )

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
        # Load directories via backend/app/config or app config
        # Let's import from backend.app.config since backend owns config & directories setup!
        try:
            from backend.app.config import OUTPUT_DIR
        except ImportError:
            # Fallback path just in case
            OUTPUT_DIR = Path("data/output")
            
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        if self.config.output_path:
            output_file = Path(self.config.output_path)
        else:
            output_file = OUTPUT_DIR / f"{audio_path.stem}_pipeline_result.json"

        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(result.to_json())

        logger.info(f"Structured output saved: {output_file.resolve()}")
