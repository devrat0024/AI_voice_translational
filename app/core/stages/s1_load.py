"""
Stage 1 — Audio Load & Validation

Validates the audio file exists and is readable.
Extracts file metadata (size, format, duration if possible).
Writes to context: 'audio_path', 'audio_metadata'
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from app.core.schemas import AudioMetadata, StageStatus
from app.core.stages.base import PipelineContext, PipelineStage


class AudioLoadStage(PipelineStage):
    name = "audio_load"
    description = "Validate audio file and extract metadata"
    optional = False

    def _execute(self, context: PipelineContext) -> tuple[StageStatus, Optional[str]]:
        audio_path: Path = context["audio_path"]

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path.resolve()}")

        size_bytes = audio_path.stat().st_size
        if size_bytes == 0:
            raise ValueError("Audio file is empty (0 bytes).")

        # Try to extract duration via pydub (non-blocking — skip if unavailable)
        duration = None
        sample_rate = None
        channels = None
        fmt = audio_path.suffix.lstrip(".").lower()

        try:
            import static_ffmpeg
            static_ffmpeg.add_paths()
            from pydub import AudioSegment
            audio = AudioSegment.from_file(audio_path)
            duration = len(audio) / 1000.0
            sample_rate = audio.frame_rate
            channels = audio.channels
        except Exception:
            pass  # Duration is optional — don't fail the stage

        metadata = AudioMetadata(
            file_path=str(audio_path.resolve()),
            file_name=audio_path.name,
            file_size_bytes=size_bytes,
            duration_seconds=duration,
            sample_rate=sample_rate,
            channels=channels,
            format=fmt,
        )

        context["audio_metadata"] = metadata
        msg = f"{audio_path.name} ({size_bytes / 1024:.1f} KB"
        msg += f", {duration:.1f}s" if duration else ""
        msg += ")"
        return StageStatus.SUCCESS, msg
