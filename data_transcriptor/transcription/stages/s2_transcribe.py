"""
Stage 2 — Whisper Speech Recognition (ASR)
"""
from __future__ import annotations

from typing import Optional

from data_transcriptor.transcription.schemas import StageStatus, TranscriptSegment
from data_transcriptor.transcription.base_stage import PipelineContext, PipelineStage


class TranscriptionStage(PipelineStage):
    name = "transcription"
    description = "Whisper automatic speech recognition"
    optional = False

    def _execute(self, context: PipelineContext) -> tuple[StageStatus, Optional[str]]:
        from data_transcriptor.transcription.speech_recognition import SpeechRecognizer

        config = context["config"]
        audio_path = context["audio_path"]

        recognizer = SpeechRecognizer(model_name=config.whisper_model)
        result = recognizer.transcribe(audio_path)

        raw_text: str = result["text"]
        segments = [
            TranscriptSegment(start=s["start"], end=s["end"], text=s["text"])
            for s in result["segments"]
        ]

        context["raw_text"] = raw_text
        context["segments"] = segments

        return StageStatus.SUCCESS, f"{len(segments)} segments, {len(raw_text)} chars"
