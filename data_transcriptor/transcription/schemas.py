"""
data_transcriptor/transcription/schemas.py — Typed Data Models for the Clinical AI Pipeline
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class PipelineConfig(BaseModel):
    whisper_model: str = Field(default="tiny", description="Whisper model size")
    ner_mode: str = Field(default="auto", description="Medical NER extraction mode")
    hf_token: str = Field(default="", description="Hugging Face token for speaker diarization")
    groq_api_key: str = Field(default="", description="Groq API key")
    groq_model: str = Field(default="llama-3.3-70b-versatile", description="Groq LLM model")
    output_path: str = Field(default="", description="Output JSON file path")
    continue_on_error: bool = Field(default=True, description="Continue pipeline on non-critical stage errors")


class StageStatus(str, Enum):
    SUCCESS = "success"
    SIMULATED = "simulated"
    FAILED = "failed"
    SKIPPED = "skipped"


class StageResult(BaseModel):
    name: str
    status: StageStatus
    duration_ms: float
    message: Optional[str] = None


class AudioMetadata(BaseModel):
    file_path: str
    file_name: str
    file_size_bytes: int
    duration_seconds: Optional[float] = None
    sample_rate: Optional[int] = None
    channels: Optional[int] = None
    format: Optional[str] = None


class TranscriptSegment(BaseModel):
    start: float
    end: float
    text: str


class DialogueTurn(BaseModel):
    speaker: str
    start: float
    end: float
    text: str


class TranscriptionResult(BaseModel):
    raw_text: str
    corrected_text: str
    segment_count: int
    dialogue: List[DialogueTurn]


class MedicalEntities(BaseModel):
    symptoms: List[str] = Field(default_factory=list)
    medicines: List[str] = Field(default_factory=list)
    primary_symptom: str = Field(default="None")
    primary_medicine: str = Field(default="None")


class SOAPNote(BaseModel):
    subjective: str
    objective: str
    assessment: str
    plan: str
    raw: str


class ClinicalIntelligenceResult(BaseModel):
    soap_note: SOAPNote
    clinical_summary: str


class PipelineMetadata(BaseModel):
    pipeline_version: str = "2.0.0"
    audio_file: str
    processed_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    total_duration_seconds: float = 0.0
    stages_run: int = 0
    stages_succeeded: int = 0
    stages_failed: int = 0
    llm_mode: str = Field(default="unknown")


class PipelineResult(BaseModel):
    metadata: PipelineMetadata
    stages: List[StageResult]
    audio: Optional[AudioMetadata] = None
    transcription: Optional[TranscriptionResult] = None
    medical_entities: Optional[MedicalEntities] = None
    clinical_intelligence: Optional[ClinicalIntelligenceResult] = None

    def to_json(self, indent: int = 2) -> str:
        return self.model_dump_json(indent=indent)

    def summary_dict(self) -> dict:
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
