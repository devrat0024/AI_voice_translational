import logging
from pathlib import Path
from .speech_recognition import SpeechRecognizer
from .diarization import SpeakerDiarizer
from .medical_ner import MedicalEntityExtractor


logger = logging.getLogger(__name__)

def align_transcript_with_speakers(whisper_segments: list, diarization_turns: list) -> list:
    """Aligns Whisper transcription segments with speaker diarization turns

    based on timestamp overlap.
    """
    aligned_dialogue = []
    
    for seg in whisper_segments:
        seg_start = seg["start"]
        seg_end = seg["end"]
        seg_text = seg["text"]
        
        # Calculate overlap with diarization speaker turns
        best_speaker = "SPEAKER_UNKNOWN"
        max_overlap = 0.0
        
        for turn in diarization_turns:
            overlap_start = max(seg_start, turn["start"])
            overlap_end = min(seg_end, turn["end"])
            overlap = max(0.0, overlap_end - overlap_start)
            
            if overlap > max_overlap:
                max_overlap = overlap
                best_speaker = turn["speaker"]
                
        # If no overlap found, assign the closest speaker
        if max_overlap == 0.0 and diarization_turns:
            # Find turn with minimum distance to segment
            closest_turn = min(diarization_turns, key=lambda t: min(abs(seg_start - t["end"]), abs(seg_end - t["start"])))
            best_speaker = closest_turn["speaker"]
            
        aligned_dialogue.append({
            "speaker": best_speaker,
            "start": seg_start,
            "end": seg_end,
            "text": seg_text
        })
        
    return aligned_dialogue

class ClinicalTranscriptionPipeline:
    def __init__(self, whisper_model: str = "tiny", ner_mode: str = "auto", hf_token: str = ""):
        """Initializes components of the clinical transcription pipeline."""
        self.recognizer = SpeechRecognizer(model_name=whisper_model)
        self.diarizer = SpeakerDiarizer(hf_token=hf_token)
        self.ner_extractor = MedicalEntityExtractor(mode=ner_mode)

    def run_pipeline(self, audio_path: Path) -> dict:
        """Executes the speech-to-structured-medical-information pipeline:

        1. Speech Recognition (Whisper)
        2. Speaker Diarization (Pyannote / Whisper Fallback)
        3. Alignment (Speaker -> Segment Text)
        4. Medical Named Entity Recognition (SciSpacy / BioBERT / Rules)
        """
        logger.info(f"Starting pipeline processing for audio: {audio_path.resolve()}")
        
        # 1. Transcribe
        transcription_result = self.recognizer.transcribe(audio_path)
        segments = transcription_result["segments"]
        full_text = transcription_result["text"]
        
        # 2. Diarize
        diarization_turns = self.diarizer.diarize(audio_path, whisper_segments=segments)
        
        # 3. Align
        aligned_dialogue = align_transcript_with_speakers(segments, diarization_turns)
        
        # 4. Extract medical entities
        structured_info = self.ner_extractor.extract_entities(full_text)
        
        logger.info("Pipeline processing complete.")
        return {
            "full_text": full_text,
            "dialogue": aligned_dialogue,
            "structured_info": structured_info
        }
