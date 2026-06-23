# AI Clinical Scribe 🩺

A unified, production-grade **Clinical AI Pipeline** that converts doctor-patient audio recordings into structured medical records.

---

## 🏗️ Architecture

```
AI_scribe/
├── run.py                    # Unified entry point (api / cli / etl)
├── requirements.txt          # All dependencies
├── app/
│   ├── config.py             # Unified configuration & .env loader
│   ├── database.py           # SQLAlchemy engine & session
│   ├── models.py             # ORM models (User, AudioRecord, ClinicalNote)
│   ├── schemas.py            # Pydantic API schemas
│   ├── auth.py               # JWT authentication
│   ├── api/
│   │   └── routes.py         # All FastAPI REST endpoints
│   ├── core/                 # ★ Structured Clinical AI Pipeline
│   │   ├── schemas.py        # Typed pipeline models (PipelineConfig, PipelineResult, …)
│   │   ├── runner.py         # ClinicalPipeline orchestrator
│   │   └── stages/           # 8 sequential pipeline stages
│   │       ├── base.py       # Abstract PipelineStage (auto-timing, error handling)
│   │       ├── s1_load.py    # Stage 1: Audio validation & metadata
│   │       ├── s2_transcribe.py  # Stage 2: Whisper ASR
│   │       ├── s3_diarize.py     # Stage 3: Speaker diarization
│   │       ├── s4_align.py       # Stage 4: Transcript-speaker alignment
│   │       ├── s5_correct.py     # Stage 5: Groq medical terminology correction
│   │       ├── s6_ner.py         # Stage 6: Medical Named Entity Recognition
│   │       ├── s7_soap.py        # Stage 7: SOAP note generation
│   │       └── s8_summary.py     # Stage 8: Clinical summary
│   ├── transcription/        # Whisper ASR + Speaker Diarization + NER + Groq LLM
│   └── pipeline/             # ETL Data Pipeline (patient CSVs, audio, documents)
└── data/                     # Runtime data (git-ignored)
```

---

## 🚀 The 8-Stage Clinical Pipeline

| # | Stage | What it does | Fallback |
|---|-------|-------------|---------|
| 1 | `audio_load` | Validate file, extract duration/sample rate | — |
| 2 | `transcription` | OpenAI Whisper ASR | — |
| 3 | `diarization` | pyannote.audio speaker ID | Simulated alternating speakers |
| 4 | `alignment` | Map each segment to its speaker via timestamp overlap | — |
| 5 | `med_correction` | Groq LLM medical spelling correction | Rules-based dictionary |
| 6 | `ner_extraction` | Medical NER (SciSpacy / BioBERT / rules) | Keyword rules |
| 7 | `soap_generation` | Groq LLM SOAP note (auto-parsed into 4 sections) | Template simulation |
| 8 | `clinical_summary` | Groq LLM concise clinical summary | Template simulation |

### Structured Output (`data/output/<audio>_pipeline_result.json`)

```json
{
  "metadata": {
    "pipeline_version": "2.0.0",
    "audio_file": "recording.mp3",
    "total_duration_seconds": 18.4,
    "llm_mode": "groq_api",
    "stages_succeeded": 8,
    "stages_failed": 0
  },
  "stages": [
    { "name": "audio_load",      "status": "success",   "duration_ms": 12 },
    { "name": "transcription",   "status": "success",   "duration_ms": 8400 },
    { "name": "diarization",     "status": "simulated", "duration_ms": 2 },
    { "name": "alignment",       "status": "success",   "duration_ms": 1 },
    { "name": "med_correction",  "status": "success",   "duration_ms": 920 },
    { "name": "ner_extraction",  "status": "success",   "duration_ms": 140 },
    { "name": "soap_generation", "status": "success",   "duration_ms": 1200 },
    { "name": "clinical_summary","status": "success",   "duration_ms": 980 }
  ],
  "transcription": {
    "raw_text": "...",
    "corrected_text": "...",
    "segment_count": 24,
    "dialogue": [
      { "speaker": "SPEAKER_01 (Doctor)", "start": 0.0, "end": 3.2, "text": "..." }
    ]
  },
  "medical_entities": {
    "symptoms": ["Fever", "Cough"],
    "medicines": ["Paracetamol"],
    "primary_symptom": "Fever",
    "primary_medicine": "Paracetamol"
  },
  "clinical_intelligence": {
    "soap_note": {
      "subjective": "...",
      "objective": "...",
      "assessment": "...",
      "plan": "...",
      "raw": "..."
    },
    "clinical_summary": "..."
  }
}
```

---

## ⚙️ Setup

### 1. Create virtual environment & install dependencies
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 2. Configure environment variables
Create a `.env` file in the project root:
```env
GROQ_API_KEY=your_groq_api_key_here
HF_TOKEN=your_huggingface_token_here     # Optional: for real speaker diarization
SECRET_KEY=your_jwt_secret_key           # For FastAPI auth
```

### 3. Install spaCy model (for medical NER)
```bash
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz
```

---

## 🖥️ Usage

### CLI — Run the full pipeline on an audio file
```bash
python run.py cli --audio path/to/recording.mp3
python run.py cli --audio recording.mp3 --whisper-model base --ner-mode rules
```

### API Server — Start the FastAPI REST server
```bash
python run.py api
# → http://127.0.0.1:8000
# → http://127.0.0.1:8000/docs  (Swagger UI)
```

### ETL — Ingest patient records, audio, and medical documents
```bash
python run.py etl init-db          # Initialize database + directories
python run.py etl run-etl          # Process all files in data/raw/
python run.py etl show-stats       # Show record counts
python run.py etl query "SELECT * FROM patients"
```

---

## 🌐 REST API Endpoints

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| `POST` | `/auth/register` | Register a new clinician | — |
| `POST` | `/auth/token` | Login → JWT token | — |
| `POST` | `/upload` | Upload clinical audio file | ✅ |
| `POST` | `/transcribe` | Whisper ASR on uploaded audio | ✅ |
| `POST` | `/ner` | Extract medical entities from text | ✅ |
| `POST` | `/summary` | Generate clinical summary | ✅ |
| `POST` | `/soap` | Generate SOAP clinical note | ✅ |
| `POST` | `/process` | Full 8-stage pipeline on uploaded audio | ✅ |

---

## 🗂️ Data Directory Layout (runtime, git-ignored)

```
data/
├── raw/
│   ├── audio/        ← Drop audio files here for ETL
│   ├── patients/     ← Drop patient CSV files here
│   └── documents/    ← Drop medical documents here
├── processed/        ← Preprocessed/normalized files
├── archive/          ← Successfully processed files
├── failed/           ← Files that failed validation
├── output/           ← Pipeline JSON results + transcripts
└── database/
    ├── clinical_scribe.db    ← API (SQLAlchemy)
    └── scribe_etl.db         ← ETL pipeline (sqlite3)
```

---

## 📦 Tech Stack

| Component | Technology |
|-----------|-----------|
| Speech Recognition | [OpenAI Whisper](https://github.com/openai/whisper) |
| Speaker Diarization | [pyannote.audio](https://github.com/pyannote/pyannote-audio) |
| Medical NER | SciSpacy / HuggingFace BioBERT / Rule-based |
| LLM Intelligence | [Groq API](https://groq.com/) (llama-3.3-70b-versatile) |
| REST API | FastAPI + Uvicorn |
| Database (API) | SQLAlchemy + SQLite |
| Database (ETL) | sqlite3 |
| Auth | JWT (python-jose) + bcrypt |
| Audio Processing | pydub + static-ffmpeg |

---

## 📝 License

MIT
