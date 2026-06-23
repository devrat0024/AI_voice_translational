"""
app/transcription/diarization.py — Speaker Diarization (pyannote.audio / simulation fallback)
"""
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class SpeakerDiarizer:
    def __init__(self, hf_token: str = ""):
        """Initializes the diarizer.

        If hf_token is provided, attempts pyannote.audio pretrained pipeline.
        Otherwise falls back to simulated alternating-speaker mode.
        """
        self.hf_token = hf_token
        self.pipeline = None

    def load_pipeline(self) -> bool:
        """Attempts to load the pyannote.audio speaker diarization pipeline."""
        if not self.hf_token:
            logger.warning(
                "HF_TOKEN not provided. Speaker diarization will run in simulation mode."
            )
            return False
        try:
            from pyannote.audio import Pipeline  # type: ignore
            logger.info("Loading pyannote.audio speaker diarization pipeline...")
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=self.hf_token,
            )
            if self.pipeline is None:
                logger.warning("pyannote pipeline returned None — using simulation mode.")
                return False
            logger.info("pyannote.audio pipeline loaded successfully.")
            return True
        except Exception as e:
            logger.warning(f"Cannot load pyannote pipeline: {e}. Using simulation mode.")
            return False

    def diarize(self, audio_path: Path, whisper_segments: list = None) -> list:
        """Diarizes the audio file.

        Returns a list of {'start': float, 'end': float, 'speaker': str} dicts.
        Falls back to simulated alternating Doctor/Patient turns when pyannote is unavailable.
        """
        has_pipeline = self.load_pipeline()

        if has_pipeline and self.pipeline is not None:
            try:
                logger.info(f"Running speaker diarization on: {audio_path.name}")
                diarization_result = self.pipeline(str(audio_path))
                turns = [
                    {"start": turn.start, "end": turn.end, "speaker": speaker}
                    for turn, _, speaker in diarization_result.itertracks(yield_label=True)
                ]
                logger.info(f"Diarization complete. {len(turns)} speaker segments found.")
                return turns
            except Exception as e:
                logger.error(f"Diarization execution failed: {e}. Falling back to simulation.")

        # Simulation fallback
        logger.info("Running simulated diarization based on transcription segments...")
        if not whisper_segments:
            return [
                {"start": 0.0, "end": 10.0, "speaker": "SPEAKER_01 (Doctor)"},
                {"start": 10.0, "end": 20.0, "speaker": "SPEAKER_02 (Patient)"},
            ]

        turns = [
            {
                "start": seg["start"],
                "end": seg["end"],
                "speaker": "SPEAKER_01 (Doctor)" if idx % 2 == 0 else "SPEAKER_02 (Patient)",
            }
            for idx, seg in enumerate(whisper_segments)
        ]
        logger.info(f"Simulated diarization complete. {len(turns)} turns created.")
        return turns
