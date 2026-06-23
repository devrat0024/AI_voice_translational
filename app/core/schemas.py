"""
app/core/schemas.py — Typed Data Models for the Clinical AI Pipeline

All inputs and outputs are fully typed Pydantic models.
The final PipelineResult serializes to a clean, structured JSON.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE CONFIGURATION (Input)
# ══════════════════════════════════════════════════════════════════════════════

class PipelineConfig(BaseModel):
    """Configuration for a single pipeline run."""

    # Whisper model size: tiny | base | small | medium | large
    whisper_model: str = Field(default="tiny", description="Whisper model size")

    # NER extraction mode: scispacy | transformer | rules | auto
    ner_mode: str = Field(default="auto", description="Medical NER extraction mode")

    # Hugging Face token (for pyannote.audio diarization)
    hf_token: str = Field(default="", description="Hugging Face token for speaker diarization")

    # Groq API key (overrides GROQ_API_KEY env var)
    groq_api_key: str = Field(default="", description="Groq API key")

    # Groq model name
    groq_model: str = Field(default="llama-3.3-70b-versatile", description="Groq LLM model")

    # Path to save the output JSON (empty = auto-generated)
    output_path: str = Field(default="", description="Output JSON file path")

    # If True, pipeline continues even if a non-critical stage fails
    continue_on_error: bool = Field(default=True, description="Continue pipeline on non-critical stage errors")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE STATUS TRACKING
# ══════════════════════════════════════════════════════════════════════════════

class StageStatus(str, Enum):
    SUCCESS = "success"
    SIMULATED = "simulated"   # Ran in fallback/simulation mode
    FAILED = "failed"
    SKIPPED = "skipped"


class StageResult(BaseModel):
    """Execution metadata for one pipeline stage."""
    name: str
    status: StageStatus
    duration_ms: float = Field(description="Wall-clock time in milliseconds")
    message: Optional[str] = Field(default=None, description="Extra info or error message")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE OUTPUT MODELS
# ══════════════════════════════════════════════════════════════════════════════

class AudioMetadata(BaseModel):
    """Audio file characteristics extracted at load time."""
    file_path: str
    file_name: str
    file_size_bytes: int
    duration_seconds: Optional[float] = None
    sample_rate: Optional[int] = None
    channels: Optional[int] = None
    format: Optional[str] = None


class TranscriptSegment(BaseModel):
    """A single Whisper-transcribed segment with timestamps."""
    start: float = Field(description="Start time in seconds")
    end: float = Field(description="End time in seconds")
    text: str


class DialogueTurn(BaseModel):
    """A speaker-aligned transcript segment."""
    speaker: str
    start: float
    end: float
    text: str


class TranscriptionResult(BaseModel):
    """Output of the transcription + diarization + alignment stages."""
    raw_text: str = Field(description="Original unmodified Whisper transcript")
    corrected_text: str = Field(description="Medically corrected transcript (Groq/rules)")
    segment_count: int
    dialogue: List[DialogueTurn]


class MedicalEntities(BaseModel):
    """Extracted medical named entities."""
    symptoms: List[str] = Field(default_factory=list)
    medicines: List[str] = Field(default_factory=list)
    # Convenience accessors — first item or 'None'
    primary_symptom: str = Field(default="None")
    primary_medicine: str = Field(default="None")


class SOAPNote(BaseModel):
    """A structured SOAP clinical note broken into its four sections."""
    subjective: str = Field(description="Patient-reported symptoms and history")
    objective: str = Field(description="Clinical observations")
    assessment: str = Field(description="Clinical assessment / diagnosis")
    plan: str = Field(description="Treatment plan")
    raw: str = Field(description="Full SOAP note text as returned by LLM")


class ClinicalIntelligenceResult(BaseModel):
    """Outputs of the LLM-powered clinical intelligence stages."""
    soap_note: SOAPNote
    clinical_summary: str


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE METADATA
# ══════════════════════════════════════════════════════════════════════════════

class PipelineMetadata(BaseModel):
    """Top-level run metadata."""
    pipeline_version: str = "2.0.0"
    audio_file: str
    processed_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    total_duration_seconds: float = 0.0
    stages_run: int = 0
    stages_succeeded: int = 0
    stages_failed: int = 0
    llm_mode: str = Field(default="unknown", description="'groq_api' or 'simulation'")


# ══════════════════════════════════════════════════════════════════════════════
# FULL PIPELINE RESULT (Final Output)
# ══════════════════════════════════════════════════════════════════════════════

class PipelineResult(BaseModel):
    """
    The complete, structured output of one clinical AI pipeline run.

    Serializes to a clean JSON with four top-level sections:
      - metadata        : run timing, version, file info
      - stages          : per-stage status and duration
      - transcription   : raw + corrected text, diarized dialogue
      - medical_entities: extracted symptoms and medicines
      - clinical_intelligence: SOAP note + clinical summary
    """
    metadata: PipelineMetadata
    stages: List[StageResult]
    audio: Optional[AudioMetadata] = None
    transcription: Optional[TranscriptionResult] = None
    medical_entities: Optional[MedicalEntities] = None
    clinical_intelligence: Optional[ClinicalIntelligenceResult] = None

    def to_json(self, indent: int = 2) -> str:
        """Serializes the full result to a JSON string."""
        return self.model_dump_json(indent=indent)

    def summary_dict(self) -> dict:
        """Returns a compact summary dict for quick display."""
        return {
            "audio_file": self.metadata.audio_file,
            "processed_at": self.metadata.processed_at,
            "total_duration_s": round(self.metadata.total_duration_seconds, 2),
            "llm_mode": self.metadata.llm_mode,
            "stages": {s.name: s.status.value for s in self.stages},
            "transcript_length": len(self.transcription.raw_text) if self.transcription else 0,
            "dialogue_turns": len(self.transcription.dialogue) if self.transcription else 0,
            "symptoms": self.medical_entities.symptoms if self.medical_entities else [],
            "medicines": self.medical_entities.medicines if self.medical_entities else [],
        }
