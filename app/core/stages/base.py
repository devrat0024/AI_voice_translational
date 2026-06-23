"""
app/core/stages/base.py — Abstract Pipeline Stage

Every stage:
  - Receives a shared, mutable PipelineContext dict
  - Returns a StageResult with status + duration
  - Has auto-timing built in via the run() wrapper
"""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict

from app.core.schemas import StageResult, StageStatus

logger = logging.getLogger(__name__)

# The pipeline context is a simple mutable dict passed through all stages
PipelineContext = Dict[str, Any]


class PipelineStage(ABC):
    """Abstract base class for all clinical pipeline stages."""

    # Override in each subclass
    name: str = "unnamed_stage"
    description: str = ""
    # If False, a failure in this stage aborts the pipeline (when continue_on_error=False)
    optional: bool = False

    def run(self, context: PipelineContext) -> StageResult:
        """Execute this stage. Times it and catches all exceptions.

        Subclasses should implement _execute(context) and return a StageStatus.
        """
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
        """Run the stage logic. Modify context in-place to pass data to next stages.

        Returns:
            (StageStatus, optional message string)
        """
        ...
