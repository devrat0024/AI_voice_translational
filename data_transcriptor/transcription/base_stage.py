"""
data_transcriptor/transcription/base_stage.py — Abstract Pipeline Stage
"""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict

from .schemas import StageResult, StageStatus

logger = logging.getLogger(__name__)

PipelineContext = Dict[str, Any]


class PipelineStage(ABC):
    name: str = "unnamed_stage"
    description: str = ""
    optional: bool = False

    def run(self, context: PipelineContext) -> StageResult:
        logger.info(f"[Stage: {self.name}] Starting — {self.description}")
        start = time.perf_counter()

        try:
            status, message = self._execute(context)
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.error(f"[Stage: {self.name}] FAILED — {exc}", exc_info=True)
            return StageResult(
                name=self.name,
                status=StageStatus.FAILED,
                duration_ms=round(elapsed_ms, 1),
                message=str(exc),
            )

        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            f"[Stage: {self.name}] {status.value.upper()} "
            f"({elapsed_ms:.0f}ms)"
            + (f" — {message}" if message else "")
        )
        return StageResult(
            name=self.name,
            status=status,
            duration_ms=round(elapsed_ms, 1),
            message=message,
        )

    @abstractmethod
    def _execute(self, context: PipelineContext) -> tuple[StageStatus, str | None]:
        ...
