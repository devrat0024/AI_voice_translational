"""
app/transcription/pipeline.py — Clinical Transcription Pipeline

Orchestrates:
  1. Whisper Speech Recognition
  2. Speaker Diarization (pyannote / simulation)
  3. Transcript-Speaker Alignment
  4. Medical Named Entity Recognition
"""
import logging
from pathlib import Path

from app.transcription.speech_recognition import SpeechRecognizer
from app.transcription.diarization import SpeakerDiarizer
from app.transcription.medical_ner import MedicalEntityExtractor

logger = logging.getLogger(__name__)


def align_transcript_with_speakers(whisper_segments: list, diarization_turns: list) -> list:
    """Aligns Whisper transcription segments with speaker diarization turns via timestamp overlap."""
    aligned_dialogue = []

    for seg in whisper_segments:
        seg_start = seg["start"]
        seg_end = seg["end"]

        best_speaker = "SPEAKER_UNKNOWN"
        max_overlap = 0.0

        for turn in diarization_turns:
            overlap = max(0.0, min(seg_end, turn["end"]) - max(seg_start, turn["start"]))
            if overlap > max_overlap:
                max_overlap = overlap
                best_speaker = turn["speaker"]

        # If no overlap, assign the nearest speaker turn
        if max_overlap == 0.0 and diarization_turns:
            closest = min(
                diarization_turns,
                key=lambda t: min(abs(seg_start - t["end"]), abs(seg_end - t["start"])),
            )
            best_speaker = closest["speaker"]

        aligned_dialogue.append({
            "speaker": best_speaker,
            "start": seg_start,
            "end": seg_end,
            "text": seg["text"],
        })

    return aligned_dialogue


class ClinicalTranscriptionPipeline:
    def __init__(self, whisper_model: str = "tiny", ner_mode: str = "auto", hf_token: str = ""):
        """Initializes all pipeline components."""
        self.recognizer = SpeechRecognizer(model_name=whisper_model)
        self.diarizer = SpeakerDiarizer(hf_token=hf_token)
        self.ner_extractor = MedicalEntityExtractor(mode=ner_mode)

    def run_pipeline(self, audio_path: Path) -> dict:
        """Executes the full speech-to-structured-medical-information pipeline.

        Steps:
          1. Speech Recognition (Whisper)
          2. Speaker Diarization (Pyannote / simulation fallback)
          3. Alignment (speaker ↔ segment text)
          4. Medical Named Entity Recognition

        Returns:
          {
            'full_text': str,
            'dialogue': list[{speaker, start, end, text}],
            'structured_info': {symptom: str, medicine: str}
          }
        """
        logger.info(f"Starting pipeline for: {audio_path.resolve()}")

        transcription_result = self.recognizer.transcribe(audio_path)
        segments = transcription_result["segments"]
        full_text = transcription_result["text"]

        diarization_turns = self.diarizer.diarize(audio_path, whisper_segments=segments)
        aligned_dialogue = align_transcript_with_speakers(segments, diarization_turns)
        structured_info = self.ner_extractor.extract_entities(full_text)

        logger.info("Pipeline processing complete.")
        return {
            "full_text": full_text,
            "dialogue": aligned_dialogue,
            "structured_info": structured_info,
        }
