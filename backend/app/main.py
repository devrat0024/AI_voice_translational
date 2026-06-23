from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from datetime import timedelta
from pathlib import Path
import os

from app.database import engine, Base, get_db
from app import models, schemas, auth
from app.services.upload import save_uploaded_file
from app.services.transcription import run_transcription_pipeline
from app.services.ner import run_ner_extraction
from app.services.summary import run_medical_correction, run_soap_generation, run_clinical_summary

# Automatically create database tables on startup
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Clinical AI Scribe Backend Services",
    description="Scalable FastAPI microservices for speech-to-structured-data, clinical NER, and SOAP generation.",
    version="1.0.0"
)

# Configure CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- AUTHENTICATION ENDPOINTS -----------------

@app.post("/auth/register", response_model=schemas.UserOut, status_code=status.HTTP_201_CREATED)
def register(user_in: schemas.UserCreate, db: Session = Depends(get_db)):
    """Registers a new user (clinician) with a hashed password."""
    db_user = db.query(models.User).filter(models.User.username == user_in.username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    hashed_pwd = auth.get_password_hash(user_in.password)
    new_user = models.User(username=user_in.username, hashed_password=hashed_pwd)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user

@app.post("/auth/token", response_model=schemas.Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    """Authenticates credentials and returns a JWT access token."""
    user = db.query(models.User).filter(models.User.username == form_data.username).first()
    if not user or not auth.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = auth.create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}

# ----------------- CLINICAL PIPELINE ENDPOINTS (PROTECTED) -----------------

@app.post("/upload", response_model=schemas.AudioRecordOut)
def upload_audio(
    file: UploadFile = File(...),
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(get_db)
):
    """Uploads a clinical audio file and registers it in the database."""
    # Validate extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in [".mp3", ".wav", ".m4a", ".ogg"]:
        raise HTTPException(status_code=400, detail="Unsupported audio file format.")
    
    try:
        saved_path = save_uploaded_file(file)
        
        # Save record in database
        audio_record = models.AudioRecord(
            filename=file.filename,
            filepath=str(saved_path.resolve()),
            owner_id=current_user.id
        )
        db.add(audio_record)
        db.commit()
        db.refresh(audio_record)
        
        return audio_record
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

@app.post("/transcribe", response_model=schemas.TranscribeResponse)
def transcribe_audio(
    audio_id: int,
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(get_db)
):
    """Executes diarized speech-to-text transcription on the registered audio file."""
    record = db.query(models.AudioRecord).filter(
        models.AudioRecord.id == audio_id,
        models.AudioRecord.owner_id == current_user.id
    ).first()
    
    if not record:
        raise HTTPException(status_code=404, detail="Audio record not found or access denied.")
    
    audio_file_path = Path(record.filepath)
    if not audio_file_path.exists():
        raise HTTPException(status_code=404, detail="Audio file missing from server storage.")
    
    try:
        pipeline_results = run_transcription_pipeline(audio_file_path)
        
        # Save transcript to DB if clinical note doesn't exist yet
        note = db.query(models.ClinicalNote).filter(models.ClinicalNote.audio_record_id == audio_id).first()
        if not note:
            note = models.ClinicalNote(
                audio_record_id=audio_id,
                transcript=pipeline_results["full_text"]
            )
            db.add(note)
        else:
            note.transcript = pipeline_results["full_text"]
        db.commit()

        # Format output
        dialogue = [
            schemas.DialogueTurn(
                speaker=turn["speaker"],
                start=turn["start"],
                end=turn["end"],
                text=turn["text"]
            ) for turn in pipeline_results["dialogue"]
        ]
        
        return schemas.TranscribeResponse(
            audio_id=audio_id,
            full_text=pipeline_results["full_text"],
            dialogue=dialogue
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

@app.post("/ner", response_model=schemas.NERResponse)
def extract_ner(
    payload: schemas.NERRequest,
    current_user: models.User = Depends(auth.get_current_user)
):
    """Extracts medical clinical entities (symptoms, drugs) from text."""
    try:
        entities = run_ner_extraction(payload.text)
        return schemas.NERResponse(
            symptom=entities.get("symptom", "None"),
            medicine=entities.get("medicine", "None")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"NER extraction failed: {str(e)}")

@app.post("/summary", response_model=schemas.SummaryResponse)
def generate_summary(
    payload: schemas.SummaryRequest,
    current_user: models.User = Depends(auth.get_current_user)
):
    """Generates a concise clinical summary from raw clinical text."""
    try:
        corrected_text = run_medical_correction(payload.text)
        summary_text = run_clinical_summary(corrected_text)
        return schemas.SummaryResponse(summary=summary_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summary generation failed: {str(e)}")

@app.post("/soap", response_model=schemas.SOAPResponse)
def generate_soap(
    payload: schemas.SOAPRequest,
    current_user: models.User = Depends(auth.get_current_user)
):
    """Generates a Subjective, Objective, Assessment, and Plan (SOAP) clinical note from text."""
    try:
        corrected_text = run_medical_correction(payload.text)
        soap_text = run_soap_generation(corrected_text)
        return schemas.SOAPResponse(soap_note=soap_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SOAP generation failed: {str(e)}")

@app.post("/process", response_model=schemas.ClinicalPipelineResultResponse)
def process_audio_end_to_end(
    audio_id: int,
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(get_db)
):
    """Runs transcription, diarization, medical NER, SOAP, and Summary end-to-end on an audio record."""
    record = db.query(models.AudioRecord).filter(
        models.AudioRecord.id == audio_id,
        models.AudioRecord.owner_id == current_user.id
    ).first()
    
    if not record:
        raise HTTPException(status_code=404, detail="Audio record not found or access denied.")
    
    audio_file_path = Path(record.filepath)
    if not audio_file_path.exists():
        raise HTTPException(status_code=404, detail="Audio file missing from server storage.")
    
    try:
        # 1. Transcribe & Diarize
        pipeline_results = run_transcription_pipeline(audio_file_path)
        full_text = pipeline_results["full_text"]
        
        # 2. Medical Correction
        corrected_text = run_medical_correction(full_text)
        
        # 3. Medical NER
        entities = run_ner_extraction(corrected_text)
        
        # 4. SOAP & Summary
        soap_text = run_soap_generation(corrected_text)
        summary_text = run_clinical_summary(corrected_text)
        
        # 5. Database Save/Update
        note = db.query(models.ClinicalNote).filter(models.ClinicalNote.audio_record_id == audio_id).first()
        if not note:
            note = models.ClinicalNote(
                audio_record_id=audio_id,
                transcript=corrected_text,
                soap_note=soap_text,
                summary=summary_text
            )
            db.add(note)
        else:
            note.transcript = corrected_text
            note.soap_note = soap_text
            note.summary = summary_text
        db.commit()

        # Format output
        dialogue = [
            schemas.DialogueTurn(
                speaker=turn["speaker"],
                start=turn["start"],
                end=turn["end"],
                text=turn["text"]
            ) for turn in pipeline_results["dialogue"]
        ]
        
        return schemas.ClinicalPipelineResultResponse(
            audio_id=audio_id,
            full_text=corrected_text,
            dialogue=dialogue,
            symptom=entities.get("symptom", "None"),
            medicine=entities.get("medicine", "None"),
            soap_note=soap_text,
            clinical_summary=summary_text
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"End-to-end processing failed: {str(e)}")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Clinical AI Scribe API. Access docs at /docs."}
