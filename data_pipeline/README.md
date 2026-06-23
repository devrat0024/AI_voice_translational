# AI Scribe Data Pipeline

A robust, modular data engineering pipeline to ingest, validate, preprocess, and store clinical audio sessions, patient demographics, and medical documents.

---

## Architecture Overview

```text
raw/                          Processed & Stored
├── patients/*.csv     ───►   Inserted into SQLite (Title Case, Standard E.164 Phones)
├── documents/*.txt    ───►   Standardized Text (Clean Whitespace) -> Saved to database
└── audio/*.mp3,*.wav  ───►   Normalized volume, resampled to 16kHz mono WAV -> Stored metadata
```

## Directory Structure

- `pipeline/`: Core modules for execution.
  - `config.py`: Directory mappings, thresholds, audio configurations.
  - `ingestion.py`: Scans directories, stages files, calculates SHA-256 checksums, and handles archiving.
  - `validation.py`: Data checks (patient CSV headers, email formats, DOB formats, audio decodability).
  - `preprocessing.py`: Audio normalization, volume leveling, resampling, E.164 phone conversions, document cleanup.
  - `storage.py`: SQLite connection managers and transactional upserts.
  - `etl.py`: Core orchestrator routing incoming data streams.
- `main.py`: CLI entry point.
- `generate_mock_data.py`: Creates mock data folder structure with test CSV files, text documents, and audio formats to verify the ETL run.

---

## Setup & Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Initialize Database and Folders**:
   This creates the SQLite database at `data/database/scribe.db` and folders like `data/raw/`, `data/processed/`, `data/archive/`, and `data/failed/`:
   ```bash
   python main.py init-db
   ```

---

## Running the Pipeline

1. **Populate Input Folders**:
   Place patient CSVs in `data/raw/patients/`, documents in `data/raw/documents/`, and audio files in `data/raw/audio/`.
   *File-Patient Association: Name files starting with the patient ID (e.g. `P1001_notes.txt`, `P1001_session.mp3`).*

2. **Run ETL Orchestration**:
   ```bash
   python main.py run-etl
   ```

3. **Check Statistics**:
   To view table row counts and processing stats:
   ```bash
   python main.py show-stats
   ```

4. **Query Data**:
   To inspect tables directly from command line:
   ```bash
   python main.py query "SELECT * FROM patients"
   python main.py query "SELECT filename, sample_rate, duration_seconds FROM audio_records"
   ```
