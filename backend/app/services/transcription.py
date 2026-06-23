"""
backend/app/services/transcription.py
LEGACY FILE — now delegates to the unified transcription package.
"""
from pathlib import Path
from backend.app.config import HF_TOKEN
from data_transcriptor.transcription.runner import ClinicalPipeline
from data_transcriptor.transcription.schemas import PipelineConfig


def run_transcription_pipeline(audio_path: Path) -> dict:
    """Executes the complete diarized transcription pipeline on the given audio file Path."""
    config = PipelineConfig(whisper_model="tiny", hf_token=HF_TOKEN)
    pipeline = ClinicalPipeline(config)
    result = pipeline.run(audio_path)
    # Convert result to dict matching the old structure if needed
    return {
        "raw_text": result.transcription.raw_text if result.transcription else "",
        "segments": [],
        "dialogue": [
            {"speaker": turn.speaker, "text": turn.text, "start": turn.start, "end": turn.end}
            for turn in (result.transcription.dialogue if result.transcription else [])
        ],
        "medical_entities": result.medical_entities.dict() if result.medical_entities else {},
        "soap_note": result.clinical_intelligence.soap_note.dict() if result.clinical_intelligence else {},
        "clinical_summary": result.clinical_intelligence.clinical_summary if result.clinical_intelligence else "",
    }
