"""
app/transcription/speech_recognition.py — Whisper ASR
"""
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class SpeechRecognizer:
    def __init__(self, model_name: str = "tiny"):
        """Initializes the Whisper model (lazy-loaded on first transcribe call)."""
        self.model_name = model_name
        self.model = None

    def load_model(self):
        """Loads the Whisper model into memory."""
        if self.model is None:
            import whisper
            logger.info(f"Loading Whisper model '{self.model_name}'...")
            self.model = whisper.load_model(self.model_name)
            logger.info("Whisper model loaded successfully.")

    def transcribe(self, audio_path: Path) -> dict:
        """Transcribes the audio file.

        Returns a dict with keys:
          - 'text': full transcript string
          - 'segments': list of {start, end, text} dicts
        """
        self.load_model()
        logger.info(f"Transcribing audio file: {audio_path.name}")

        import torch
        fp16_val = torch.cuda.is_available()
        result = self.model.transcribe(str(audio_path), fp16=fp16_val)

        segments = [
            {"start": seg["start"], "end": seg["end"], "text": seg["text"].strip()}
            for seg in result.get("segments", [])
        ]
        logger.info(f"Transcription complete. {len(segments)} segments produced.")
        return {"text": result.get("text", "").strip(), "segments": segments}
