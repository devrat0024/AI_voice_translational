from pydantic import BaseModel
from datetime import datetime
from typing import List, Optional

# Authentication Schemas
class UserBase(BaseModel):
    username: str

class UserCreate(UserBase):
    password: str

class UserOut(UserBase):
    id: int
    
    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

# Audio Storage Schemas
class AudioRecordOut(BaseModel):
    id: int
    filename: str
    uploaded_at: datetime
    owner_id: int

    class Config:
        from_attributes = True

# Dialogue Schema for transcription
class DialogueTurn(BaseModel):
    speaker: str
    start: float
    end: float
    text: str

# API Service Payload Schemas
class TranscribeResponse(BaseModel):
    audio_id: int
    full_text: str
    dialogue: List[DialogueTurn]

class NERRequest(BaseModel):
    text: str

class NERResponse(BaseModel):
    symptom: str
    medicine: str

class SummaryRequest(BaseModel):
    text: str

class SummaryResponse(BaseModel):
    summary: str

class SOAPRequest(BaseModel):
    text: str

class SOAPResponse(BaseModel):
    soap_note: str

# Complete end-to-end processing output schema
class ClinicalPipelineResultResponse(BaseModel):
    audio_id: int
    full_text: str
    dialogue: List[DialogueTurn]
    symptom: str
    medicine: str
    soap_note: str
    clinical_summary: str
