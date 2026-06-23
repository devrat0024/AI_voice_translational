"""
app/pipeline/preprocessing.py — Audio & Document Preprocessing for ETL Pipeline
"""
import re
import json
import logging
from pathlib import Path

import pandas as pd

from app.pipeline.config import (
    PROCESSED_AUDIO_DIR,
    PROCESSED_DOCUMENTS_DIR,
    TARGET_SAMPLE_RATE,
    TARGET_CHANNELS,
    TARGET_FORMAT,
)

logger = logging.getLogger(__name__)


def preprocess_audio(file_path: Path) -> dict:
    """Preprocesses a raw audio file:

    - Converts to mono WAV.
    - Resamples to 16 kHz.
    - Normalizes volume.
    - Saves to data/processed/audio/.

    Returns a dict of audio characteristics.
    """
    import static_ffmpeg
    static_ffmpeg.add_paths()
    from pydub import AudioSegment
    from pydub.effects import normalize

    audio = AudioSegment.from_file(file_path)
    normalized_audio = normalize(audio)
    processed_audio = normalized_audio.set_frame_rate(TARGET_SAMPLE_RATE).set_channels(TARGET_CHANNELS)

    output_filename = f"{file_path.stem}_processed.wav"
    output_path = PROCESSED_AUDIO_DIR / output_filename
    processed_audio.export(output_path, format=TARGET_FORMAT)
    logger.info(f"Preprocessed audio saved to {output_path}")

    return {
        "file_path": str(output_path),
        "filename": output_filename,
        "duration_seconds": len(processed_audio) / 1000.0,
        "sample_rate": TARGET_SAMPLE_RATE,
        "channels": TARGET_CHANNELS,
        "file_size_bytes": output_path.stat().st_size,
    }


def clean_phone_number(phone) -> str:
    """Cleans phone numbers into a standardized digits-only format."""
    if pd.isna(phone):
        return ""
    phone_str = str(phone).strip()
    has_plus = phone_str.startswith("+")
    digits = re.sub(r"\D", "", phone_str)
    if digits:
        return f"+{digits}" if has_plus else digits
    return ""


def preprocess_patient_row(row: pd.Series) -> dict:
    """Standardizes patient record fields (Title Case names, clean emails, E.164 phones)."""
    return {
        "patient_id": str(row["patient_id"]).strip(),
        "first_name": str(row["first_name"]).strip().title(),
        "last_name": str(row["last_name"]).strip().title(),
        "dob": str(row["dob"]).strip(),
        "gender": str(row["gender"]).strip().capitalize() if pd.notna(row.get("gender")) else "Unknown",
        "email": str(row["email"]).strip().lower() if pd.notna(row.get("email")) else "",
        "phone": clean_phone_number(row.get("phone")),
    }


def preprocess_medical_document(file_path: Path) -> dict:
    """Cleans and extracts text from a medical document.

    Supports .txt, .md, .json, .pdf.
    Saves a clean copy to data/processed/documents/.
    Returns a dict with file_path, filename, document_type, content.
    """
    ext = file_path.suffix.lower()
    content = ""

    if ext in {".txt", ".md"}:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    elif ext == ".json":
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                content = json.dumps(data, indent=2)
            except Exception:
                f.seek(0)
                content = f.read()
    elif ext == ".pdf":
        try:
            import pypdf
            reader = pypdf.PdfReader(file_path)
            content = "\n".join(page.extract_text() for page in reader.pages)
        except ImportError:
            content = f"[PDF contents of {file_path.name} — install pypdf for extraction]"

    # Normalize whitespace
    content_clean = re.sub(r"\r\n", "\n", content)
    content_clean = re.sub(r"[ \t]+", " ", content_clean).strip()

    # Infer document type from filename
    doc_type = "clinical_note"
    filename_lower = file_path.name.lower()
    if "summary" in filename_lower:
        doc_type = "summary"
    elif "report" in filename_lower or "lab" in filename_lower:
        doc_type = "lab_report"

    # Override from first-line header if present
    for line in content_clean.split("\n")[:5]:
        match = re.match(r"^(document\s*type|type)\s*:\s*(.+)$", line, re.IGNORECASE)
        if match:
            doc_type = match.group(2).strip().lower().replace(" ", "_")
            break

    # Save processed copy
    output_filename = f"{file_path.stem}_processed.txt"
    output_path = PROCESSED_DOCUMENTS_DIR / output_filename
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content_clean)

    return {
        "file_path": str(output_path),
        "filename": output_filename,
        "document_type": doc_type,
        "content": content_clean,
    }
