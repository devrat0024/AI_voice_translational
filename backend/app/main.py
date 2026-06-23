"""
backend/app/main.py — FastAPI Application & Route Definitions
"""
import shutil
import logging
from pathlib import Path

from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from backend.app.database import engine, Base, get_db
from backend.app import models, schemas, auth
from backend.app.config import UPLOAD_DIR, HF_TOKEN
from backend.app.services.encryption import encrypt_data, decrypt_data
from data_transcriptor.transcription.medical_ner import MedicalEntityExtractor
from data_transcriptor.transcription.llm_layer import ClinicalIntelligenceLayer
from data_transcriptor.transcription.pipeline import ClinicalTranscriptionPipeline
from data_transcriptor.transcription.runner import ClinicalPipeline
from data_transcriptor.transcription.schemas import PipelineConfig

logger = logging.getLogger(__name__)

# Try to get GROQ_MODEL from config
try:
    from backend.app.config import GROQ_MODEL
except ImportError:
    GROQ_MODEL = "llama-3.3-70b-versatile"

# Singleton AI components loaded once at startup
_ner_extractor = MedicalEntityExtractor(mode="auto")
_intel_layer = ClinicalIntelligenceLayer(model_name=GROQ_MODEL)


def log_audit(db: Session, user_id: int | None, action: str, resource: str | None = None, details: str | None = None):
    """Utility to persist security audit events to the database."""
    try:
        audit_entry = models.AuditLog(
            user_id=user_id,
            action=action,
            resource=resource,
            details=details
        )
        db.add(audit_entry)
        db.commit()
    except Exception as e:
        logger.error(f"Failed to record audit log: {e}")


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


@app.on_event("startup")
async def on_startup():
    """Initializes directories and creates DB tables on startup."""
    from backend.app.config import load_env_file
    load_env_file()
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(bind=engine)
    logger.info("Startup complete. Database tables verified/initialized.")


@app.get("/", tags=["Health"])
def read_root():
    return {"message": "Welcome to the Clinical AI Scribe API. Visit /docs for the Swagger UI."}


# ── Authentication ────────────────────────────────────────────────────────────
@app.post("/auth/register", response_model=schemas.UserOut, status_code=status.HTTP_201_CREATED, tags=["Auth"])
def register(user_in: schemas.UserCreate, db: Session = Depends(get_db)):
    """Registers a new clinician account with a bcrypt-hashed password."""
    db_user = db.query(models.User).filter(models.User.username == user_in.username).first()
    if db_user:
        log_audit(db, None, "REGISTER_FAILED", details=f"Attempted username: {user_in.username}")
        raise HTTPException(status_code=400, detail="Username already registered")
    hashed_pwd = auth.get_password_hash(user_in.password)
    new_user = models.User(username=user_in.username, hashed_password=hashed_pwd, role=user_in.role or "doctor")
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    log_audit(db, new_user.id, "REGISTER_SUCCESS", resource=f"User:{new_user.id}", details=f"Registered role: {new_user.role}")
    return new_user


@app.post("/auth/token", response_model=schemas.Token, tags=["Auth"])
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    """Authenticates credentials and returns a signed JWT access token."""
    user = db.query(models.User).filter(models.User.username == form_data.username).first()
    if not user or not auth.verify_password(form_data.password, user.hashed_password):
        log_audit(db, None, "LOGIN_FAILED", details=f"Failed login attempt for: {form_data.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = auth.create_access_token(data={"sub": user.username})
    log_audit(db, user.id, "LOGIN_SUCCESS", resource=f"User:{user.id}", details=f"Login successful for role: {user.role}")
    return {"access_token": access_token, "token_type": "bearer"}



# ── Audio Upload ──────────────────────────────────────────────────────────────
@app.post("/upload", response_model=schemas.AudioRecordOut, tags=["Clinical Pipeline"])
def upload_audio(
    file: UploadFile = File(...),
    current_user: models.User = Depends(auth.RoleChecker(["doctor", "admin"])),
    db: Session = Depends(get_db),
):
    """Uploads a clinical audio file and registers it in the database."""
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in {".mp3", ".wav", ".m4a", ".ogg"}:
        log_audit(db, current_user.id, "UPLOAD_FAILED", details=f"Unsupported format: {file.filename}")
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
        log_audit(db, current_user.id, "UPLOAD_SUCCESS", resource=f"AudioRecord:{audio_record.id}", details=f"Uploaded: {file.filename}")
        return audio_record
    except Exception as e:
        log_audit(db, current_user.id, "UPLOAD_ERROR", details=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")


# ── Transcription ─────────────────────────────────────────────────────────────
@app.post("/transcribe", response_model=schemas.TranscribeResponse, tags=["Clinical Pipeline"])
def transcribe_audio(
    audio_id: int,
    current_user: models.User = Depends(auth.RoleChecker(["doctor", "admin"])),
    db: Session = Depends(get_db),
):
    """Runs diarized Whisper speech-to-text on the registered audio file."""
    record = db.query(models.AudioRecord).filter(
        models.AudioRecord.id == audio_id,
        models.AudioRecord.owner_id == current_user.id,
    ).first()
    if not record:
        log_audit(db, current_user.id, "TRANSCRIBE_UNAUTHORIZED", resource=f"AudioRecord:{audio_id}")
        raise HTTPException(status_code=404, detail="Audio record not found or access denied.")

    audio_file_path = Path(record.filepath)
    if not audio_file_path.exists():
        log_audit(db, current_user.id, "TRANSCRIBE_FAILED", resource=f"AudioRecord:{audio_id}", details="Audio file missing")
        raise HTTPException(status_code=404, detail="Audio file missing from server storage.")

    try:
        pipeline = ClinicalTranscriptionPipeline(whisper_model="tiny", hf_token=HF_TOKEN)
        pipeline_results = pipeline.run_pipeline(audio_file_path)

        note = db.query(models.ClinicalNote).filter(
            models.ClinicalNote.audio_record_id == audio_id
        ).first()
        
        # Encrypt the transcript value before storing in DB
        encrypted_transcript = encrypt_data(pipeline_results["full_text"])
        
        if not note:
            note = models.ClinicalNote(audio_record_id=audio_id, transcript=encrypted_transcript)
            db.add(note)
        else:
            note.transcript = encrypted_transcript
        db.commit()

        log_audit(db, current_user.id, "TRANSCRIBE_SUCCESS", resource=f"AudioRecord:{audio_id}")

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
        log_audit(db, current_user.id, "TRANSCRIBE_ERROR", resource=f"AudioRecord:{audio_id}", details=str(e))
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


# ── Medical NER ───────────────────────────────────────────────────────────────
@app.post("/ner", response_model=schemas.NERResponse, tags=["Clinical Pipeline"])
def extract_ner(
    payload: schemas.NERRequest,
    current_user: models.User = Depends(auth.RoleChecker(["doctor", "admin"])),
    db: Session = Depends(get_db),
):
    """Extracts medical entities (symptoms, drugs) from clinical text."""
    try:
        entities = _ner_extractor.extract_entities(payload.text)
        log_audit(db, current_user.id, "NER_EXTRACTION_SUCCESS")
        return schemas.NERResponse(
            symptom=entities.get("symptom", "None"),
            medicine=entities.get("medicine", "None"),
        )
    except Exception as e:
        log_audit(db, current_user.id, "NER_EXTRACTION_ERROR", details=str(e))
        raise HTTPException(status_code=500, detail=f"NER extraction failed: {str(e)}")


# ── Clinical Summary ──────────────────────────────────────────────────────────
@app.post("/summary", response_model=schemas.SummaryResponse, tags=["Clinical Pipeline"])
def generate_summary(
    payload: schemas.SummaryRequest,
    current_user: models.User = Depends(auth.RoleChecker(["doctor", "admin"])),
    db: Session = Depends(get_db),
):
    """Generates a concise clinical summary from raw clinical text."""
    try:
        corrected = _intel_layer.medical_correction(payload.text)
        summary = _intel_layer.generate_clinical_summary(corrected)
        log_audit(db, current_user.id, "SUMMARY_GENERATION_SUCCESS")
        return schemas.SummaryResponse(summary=summary)
    except Exception as e:
        log_audit(db, current_user.id, "SUMMARY_GENERATION_ERROR", details=str(e))
        raise HTTPException(status_code=500, detail=f"Summary generation failed: {str(e)}")


# ── SOAP Note ─────────────────────────────────────────────────────────────────
@app.post("/soap", response_model=schemas.SOAPResponse, tags=["Clinical Pipeline"])
def generate_soap(
    payload: schemas.SOAPRequest,
    current_user: models.User = Depends(auth.RoleChecker(["doctor", "admin"])),
    db: Session = Depends(get_db),
):
    """Generates a SOAP (Subjective, Objective, Assessment, Plan) clinical note."""
    try:
        corrected = _intel_layer.medical_correction(payload.text)
        soap = _intel_layer.generate_soap_note(corrected)
        log_audit(db, current_user.id, "SOAP_GENERATION_SUCCESS")
        return schemas.SOAPResponse(soap_note=soap)
    except Exception as e:
        log_audit(db, current_user.id, "SOAP_GENERATION_ERROR", details=str(e))
        raise HTTPException(status_code=500, detail=f"SOAP generation failed: {str(e)}")


# ── End-to-End Processing ─────────────────────────────────────────────────────
@app.post("/process", response_model=schemas.ClinicalPipelineResultResponse, tags=["Clinical Pipeline"])
def process_audio_end_to_end(
    audio_id: int,
    whisper_model: str = "tiny",
    current_user: models.User = Depends(auth.RoleChecker(["doctor", "admin"])),
    db: Session = Depends(get_db),
):
    """Runs the full structured 8-stage pipeline on an audio record with AES encrypted storage."""
    record = db.query(models.AudioRecord).filter(
        models.AudioRecord.id == audio_id,
        models.AudioRecord.owner_id == current_user.id,
    ).first()
    if not record:
        log_audit(db, current_user.id, "PROCESS_UNAUTHORIZED", resource=f"AudioRecord:{audio_id}")
        raise HTTPException(status_code=404, detail="Audio record not found or access denied.")

    audio_file_path = Path(record.filepath)
    if not audio_file_path.exists():
        log_audit(db, current_user.id, "PROCESS_FAILED", resource=f"AudioRecord:{audio_id}", details="Audio file missing")
        raise HTTPException(status_code=404, detail="Audio file missing from server storage.")

    try:
        config = PipelineConfig(whisper_model=whisper_model, hf_token=HF_TOKEN)
        pipeline = ClinicalPipeline(config)
        result = pipeline.run(audio_file_path)

        # Retrieve plain texts
        corrected_text = result.transcription.corrected_text if result.transcription else ""
        soap_text = result.clinical_intelligence.soap_note.raw if result.clinical_intelligence else ""
        summary_text = result.clinical_intelligence.clinical_summary if result.clinical_intelligence else ""

        # Encrypt the text fields prior to database persistence
        enc_corrected = encrypt_data(corrected_text)
        enc_soap = encrypt_data(soap_text)
        enc_summary = encrypt_data(summary_text)

        note = db.query(models.ClinicalNote).filter(
            models.ClinicalNote.audio_record_id == audio_id
        ).first()
        if not note:
            note = models.ClinicalNote(
                audio_record_id=audio_id,
                transcript=enc_corrected,
                soap_note=enc_soap,
                summary=enc_summary,
            )
            db.add(note)
        else:
            note.transcript = enc_corrected
            note.soap_note = enc_soap
            note.summary = enc_summary
        db.commit()

        log_audit(db, current_user.id, "PROCESS_SUCCESS", resource=f"AudioRecord:{audio_id}")

        # Map to response schema (returning plain, unencrypted text to authorized user)
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
        log_audit(db, current_user.id, "PROCESS_ERROR", resource=f"AudioRecord:{audio_id}", details=str(e))
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {str(e)}")


# ── Audit Trails (Admin Only) ──────────────────────────────────────────────────
@app.get("/admin/audit-logs", response_model=list[schemas.AuditLogOut], tags=["Admin"])
def get_audit_logs(
    current_user: models.User = Depends(auth.RoleChecker(["admin"])),
    db: Session = Depends(get_db),
):
    """Retrieves security audit logs from the database. Restricted to administrator role only."""
    logs = db.query(models.AuditLog).order_by(models.AuditLog.timestamp.desc()).all()
    log_audit(db, current_user.id, "VIEW_AUDIT_LOGS", details=f"Fetched {len(logs)} audit entries.")
    return logs

