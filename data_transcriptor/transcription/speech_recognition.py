import logging
import whisper
from pathlib import Path

logger = logging.getLogger(__name__)

class SpeechRecognizer:
    def __init__(self, model_name: str = "tiny"):
        """Initializes the Whisper model."""
        self.model_name = model_name
        self.model = None

    def load_model(self):
        """Loads the Whisper model into memory."""
        if self.model is None:
            logger.info(f"Loading Whisper model '{self.model_name}'...")
            # Whisper handles FP16 CPU warning internally; we'll catch it or run with standard params
            self.model = whisper.load_model(self.model_name)
            logger.info("Whisper model loaded successfully.")

    def transcribe(self, audio_path: Path) -> dict:
        """Transcribes the audio file and returns a dictionary with consolidated text

        and timestamped segments.
        """
        self.load_model()
        
        logger.info(f"Transcribing audio file: {audio_path.name}")
        # FP16 is not supported on CPU, so we default to FP32 if running on CPU
        # We check torch.cuda.is_available() to determine if gpu is available
        import torch
        fp16_val = torch.cuda.is_available()
        
        result = self.model.transcribe(str(audio_path), fp16=fp16_val)
        
        segments = []
        for seg in result.get("segments", []):
            segments.append({
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"].strip()
            })
            
        logger.info(f"Transcription complete. Transcribed {len(segments)} segments.")
        return {
            "text": result.get("text", "").strip(),
            "segments": segments
        }
