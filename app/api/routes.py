"""
app/api/routes.py — FastAPI Application & All Route Definitions

Merged from backend/app/main.py and backend/app/services/*.py
All logic is inlined here; no intermediate service proxy files needed.
"""
import shutil
import logging
from pathlib import Path

from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from app.database import engine, Base, get_db
from app import models, schemas, auth
from app.config import UPLOAD_DIR, GROQ_MODEL, HF_TOKEN
from app.transcription.pipeline import ClinicalTranscriptionPipeline
from app.transcription.medical_ner import MedicalEntityExtractor
from app.transcription.llm_layer import ClinicalIntelligenceLayer

logger = logging.getLogger(__name__)


# ── Singleton AI components (loaded once at startup) ─────────────────────────
_ner_extractor = MedicalEntityExtractor(mode="auto")
_intel_layer = ClinicalIntelligenceLayer(model_name=GROQ_MODEL)

# ── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Clinical AI Scribe API",
    description=(
        "Unified API for speech-to-structured-data, clinical NER, "
        "SOAP note generation, and ETL data pipeline management."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Startup Event ────────────────────────────────────────────────────────────
@app.on_event("startup")
async def on_startup():
    """Initializes directories and creates DB tables on first run."""
    from app.config import init_directories
    init_directories()
    Base.metadata.create_all(bind=engine)
    logger.info("Startup complete. Directories and database tables initialized.")

@app.get("/", tags=["Health"])
def read_root():
    return {"message": "Welcome to the Clinical AI Scribe API. Visit /docs for the Swagger UI."}


# ── Authentication ────────────────────────────────────────────────────────────
@app.post("/auth/register", response_model=schemas.UserOut, status_code=status.HTTP_201_CREATED, tags=["Auth"])
def register(user_in: schemas.UserCreate, db: Session = Depends(get_db)):
    """Registers a new clinician account with a bcrypt-hashed password."""
    db_user = db.query(models.User).filter(models.User.username == user_in.username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    hashed_pwd = auth.get_password_hash(user_in.password)
    new_user = models.User(username=user_in.username, hashed_password=hashed_pwd)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user


@app.post("/auth/token", response_model=schemas.Token, tags=["Auth"])
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    """Authenticates credentials and returns a signed JWT access token."""
    user = db.query(models.User).filter(models.User.username == form_data.username).first()
    if not user or not auth.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = auth.create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}


# ── Audio Upload ──────────────────────────────────────────────────────────────
@app.post("/upload", response_model=schemas.AudioRecordOut, tags=["Clinical Pipeline"])
def upload_audio(
    file: UploadFile = File(...),
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(get_db),
):
    """Uploads a clinical audio file and registers it in the database."""
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in {".mp3", ".wav", ".m4a", ".ogg"}:
        raise HTTPException(status_code=400, detail="Unsupported audio file format.")
    try:
        destination = UPLOAD_DIR / file.filename
        with destination.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        audio_record = models.AudioRecord(
            filename=file.filename,
            filepath=str(destination.resolve()),
            owner_id=current_user.id,
        )
        db.add(audio_record)
        db.commit()
        db.refresh(audio_record)
        return audio_record
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")


# ── Transcription ─────────────────────────────────────────────────────────────
@app.post("/transcribe", response_model=schemas.TranscribeResponse, tags=["Clinical Pipeline"])
def transcribe_audio(
    audio_id: int,
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(get_db),
):
    """Runs diarized Whisper speech-to-text on the registered audio file."""
    record = db.query(models.AudioRecord).filter(
        models.AudioRecord.id == audio_id,
        models.AudioRecord.owner_id == current_user.id,
    ).first()
    if not record:
        raise HTTPException(status_code=404, detail="Audio record not found or access denied.")

    audio_file_path = Path(record.filepath)
    if not audio_file_path.exists():
        raise HTTPException(status_code=404, detail="Audio file missing from server storage.")

    try:
        pipeline = ClinicalTranscriptionPipeline(whisper_model="tiny", hf_token=HF_TOKEN)
        pipeline_results = pipeline.run_pipeline(audio_file_path)

        note = db.query(models.ClinicalNote).filter(
            models.ClinicalNote.audio_record_id == audio_id
        ).first()
        if not note:
            note = models.ClinicalNote(audio_record_id=audio_id, transcript=pipeline_results["full_text"])
            db.add(note)
        else:
            note.transcript = pipeline_results["full_text"]
        db.commit()

        dialogue = [
            schemas.DialogueTurn(
                speaker=turn["speaker"], start=turn["start"], end=turn["end"], text=turn["text"]
            )
            for turn in pipeline_results["dialogue"]
        ]
        return schemas.TranscribeResponse(
            audio_id=audio_id, full_text=pipeline_results["full_text"], dialogue=dialogue
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


# ── Medical NER ───────────────────────────────────────────────────────────────
@app.post("/ner", response_model=schemas.NERResponse, tags=["Clinical Pipeline"])
def extract_ner(
    payload: schemas.NERRequest,
    current_user: models.User = Depends(auth.get_current_user),
):
    """Extracts medical entities (symptoms, drugs) from clinical text."""
    try:
        entities = _ner_extractor.extract_entities(payload.text)
        return schemas.NERResponse(
            symptom=entities.get("symptom", "None"),
            medicine=entities.get("medicine", "None"),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"NER extraction failed: {str(e)}")


# ── Clinical Summary ──────────────────────────────────────────────────────────
@app.post("/summary", response_model=schemas.SummaryResponse, tags=["Clinical Pipeline"])
def generate_summary(
    payload: schemas.SummaryRequest,
    current_user: models.User = Depends(auth.get_current_user),
):
    """Generates a concise clinical summary from raw clinical text."""
    try:
        corrected = _intel_layer.medical_correction(payload.text)
        summary = _intel_layer.generate_clinical_summary(corrected)
        return schemas.SummaryResponse(summary=summary)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summary generation failed: {str(e)}")


# ── SOAP Note ─────────────────────────────────────────────────────────────────
@app.post("/soap", response_model=schemas.SOAPResponse, tags=["Clinical Pipeline"])
def generate_soap(
    payload: schemas.SOAPRequest,
    current_user: models.User = Depends(auth.get_current_user),
):
    """Generates a SOAP (Subjective, Objective, Assessment, Plan) clinical note."""
    try:
        corrected = _intel_layer.medical_correction(payload.text)
        soap = _intel_layer.generate_soap_note(corrected)
        return schemas.SOAPResponse(soap_note=soap)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SOAP generation failed: {str(e)}")


# ── End-to-End Processing ─────────────────────────────────────────────────────
@app.post("/process", response_model=schemas.ClinicalPipelineResultResponse, tags=["Clinical Pipeline"])
def process_audio_end_to_end(
    audio_id: int,
    whisper_model: str = "tiny",
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(get_db),
):
    """Runs the full structured 8-stage pipeline on an audio record.

    Stages: audio_load → transcription → diarization → alignment →
            med_correction → ner_extraction → soap_generation → clinical_summary
    """
    from app.core.runner import ClinicalPipeline
    from app.core.schemas import PipelineConfig

    record = db.query(models.AudioRecord).filter(
        models.AudioRecord.id == audio_id,
        models.AudioRecord.owner_id == current_user.id,
    ).first()
    if not record:
        raise HTTPException(status_code=404, detail="Audio record not found or access denied.")

    audio_file_path = Path(record.filepath)
    if not audio_file_path.exists():
        raise HTTPException(status_code=404, detail="Audio file missing from server storage.")

    try:
        config = PipelineConfig(whisper_model=whisper_model, hf_token=HF_TOKEN)
        pipeline = ClinicalPipeline(config)
        result = pipeline.run(audio_file_path)

        # Persist transcript & SOAP to DB
        corrected_text = result.transcription.corrected_text if result.transcription else ""
        soap_text = result.clinical_intelligence.soap_note.raw if result.clinical_intelligence else ""
        summary_text = result.clinical_intelligence.clinical_summary if result.clinical_intelligence else ""

        note = db.query(models.ClinicalNote).filter(
            models.ClinicalNote.audio_record_id == audio_id
        ).first()
        if not note:
            note = models.ClinicalNote(
                audio_record_id=audio_id,
                transcript=corrected_text,
                soap_note=soap_text,
                summary=summary_text,
            )
            db.add(note)
        else:
            note.transcript = corrected_text
            note.soap_note = soap_text
            note.summary = summary_text
        db.commit()

        # Map to response schema
        dialogue = [
            schemas.DialogueTurn(speaker=t.speaker, start=t.start, end=t.end, text=t.text)
            for t in (result.transcription.dialogue if result.transcription else [])
        ]
        entities = result.medical_entities

        return schemas.ClinicalPipelineResultResponse(
            audio_id=audio_id,
            full_text=corrected_text,
            dialogue=dialogue,
            symptom=entities.primary_symptom if entities else "None",
            medicine=entities.primary_medicine if entities else "None",
            soap_note=soap_text,
            clinical_summary=summary_text,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {str(e)}")
