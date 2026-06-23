"""
Stage 3 — Speaker Diarization

Identifies speaker turns using pyannote.audio (or simulates alternating speakers).
Writes to context: 'diarization_turns', 'diarization_simulated'
"""
from __future__ import annotations

from typing import Optional

from app.core.schemas import StageStatus
from app.core.stages.base import PipelineContext, PipelineStage


class DiarizationStage(PipelineStage):
    name = "diarization"
    description = "Speaker identification and turn detection"
    optional = True  # Falls back gracefully to simulation

    def _execute(self, context: PipelineContext) -> tuple[StageStatus, Optional[str]]:
        from app.transcription.diarization import SpeakerDiarizer

        config = context["config"]
        audio_path = context["audio_path"]
        segments = context.get("segments", [])

        # Convert TranscriptSegment models back to dicts for the diarizer
        seg_dicts = [{"start": s.start, "end": s.end, "text": s.text} for s in segments]

        diarizer = SpeakerDiarizer(hf_token=config.hf_token)
        simulated = not bool(config.hf_token)  # No token = simulation mode

        turns = diarizer.diarize(audio_path, whisper_segments=seg_dicts)
        context["diarization_turns"] = turns
        context["diarization_simulated"] = simulated

        status = StageStatus.SIMULATED if simulated else StageStatus.SUCCESS
        msg = f"{len(turns)} turns ({'simulated' if simulated else 'pyannote'})"
        return status, msg
