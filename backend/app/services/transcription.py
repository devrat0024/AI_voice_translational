import sys
from pathlib import Path
from app.config import TRANSCRIPTOR_DIR

# Ensure transcription package imports work
if str(TRANSCRIPTOR_DIR) not in sys.path:
    sys.path.append(str(TRANSCRIPTOR_DIR))

from transcription.pipeline import ClinicalTranscriptionPipeline
from transcription.config import HF_TOKEN

def run_transcription_pipeline(audio_path: Path) -> dict:
    """Executes the complete diarized transcription pipeline on the given audio file Path."""
    pipeline = ClinicalTranscriptionPipeline(whisper_model="tiny", hf_token=HF_TOKEN)
    return pipeline.run_pipeline(audio_path)
