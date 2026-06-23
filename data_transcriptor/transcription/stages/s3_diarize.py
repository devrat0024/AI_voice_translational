"""
Stage 3 — Speaker Diarization
"""
from __future__ import annotations

from typing import Optional

from data_transcriptor.transcription.schemas import StageStatus
from data_transcriptor.transcription.base_stage import PipelineContext, PipelineStage


class DiarizationStage(PipelineStage):
    name = "diarization"
    description = "Speaker identification and turn detection"
    optional = True

    def _execute(self, context: PipelineContext) -> tuple[StageStatus, Optional[str]]:
        from data_transcriptor.transcription.diarization import SpeakerDiarizer

        config = context["config"]
        audio_path = context["audio_path"]
        segments = context.get("segments", [])

        seg_dicts = [{"start": s.start, "end": s.end, "text": s.text} for s in segments]

        diarizer = SpeakerDiarizer(hf_token=config.hf_token)
        simulated = not bool(config.hf_token)

        turns = diarizer.diarize(audio_path, whisper_segments=seg_dicts)
        context["diarization_turns"] = turns
        context["diarization_simulated"] = simulated

        status = StageStatus.SIMULATED if simulated else StageStatus.SUCCESS
        msg = f"{len(turns)} turns ({'simulated' if simulated else 'pyannote'})"
        return status, msg
