"""
backend/app/services/transcription.py
LEGACY FILE — now delegates to the unified app package.
"""
from pathlib import Path
from app.transcription.pipeline import ClinicalTranscriptionPipeline
from app.config import HF_TOKEN


def run_transcription_pipeline(audio_path: Path) -> dict:
    """Executes the complete diarized transcription pipeline on the given audio file Path."""
    pipeline = ClinicalTranscriptionPipeline(whisper_model="tiny", hf_token=HF_TOKEN)
    return pipeline.run_pipeline(audio_path)
