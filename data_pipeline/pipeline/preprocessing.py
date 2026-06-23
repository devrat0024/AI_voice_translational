import re
import logging
from pathlib import Path
import pandas as pd
from pydub import AudioSegment
from pydub.effects import normalize
from .config import (
    PROCESSED_AUDIO_DIR,
    PROCESSED_DOCUMENTS_DIR,
    TARGET_SAMPLE_RATE,
    TARGET_CHANNELS,
    TARGET_FORMAT
)

logger = logging.getLogger(__name__)

def preprocess_audio(file_path: Path) -> dict:
    """Preprocesses a raw audio file:

    - Converts to mono WAV format.
    - Resamples to 16,000 Hz.
    - Normalizes volume.
    - Saves to processed/audio/.
    Returns a dictionary of audio characteristics.
    """
    import static_ffmpeg
    static_ffmpeg.add_paths()

    # Load audio
    audio = AudioSegment.from_file(file_path)

    # Normalize audio volume levels
    normalized_audio = normalize(audio)

    # Convert to mono (1 channel) and target sample rate (16kHz)
    processed_audio = normalized_audio.set_frame_rate(TARGET_SAMPLE_RATE).set_channels(TARGET_CHANNELS)

    # Output path
    output_filename = f"{file_path.stem}_processed.wav"
    output_path = PROCESSED_AUDIO_DIR / output_filename

    # Export
    processed_audio.export(output_path, format=TARGET_FORMAT)
    logger.info(f"Preprocessed audio saved to {output_path}")

    return {
        "file_path": str(output_path),
        "filename": output_filename,
        "duration_seconds": len(processed_audio) / 1000.0,
        "sample_rate": TARGET_SAMPLE_RATE,
        "channels": TARGET_CHANNELS,
        "file_size_bytes": output_path.stat().st_size
    }

def clean_phone_number(phone) -> str:
    """Cleans phone numbers into a standardized E.164-like format (digits only, maintaining leading +)."""
    if pd.isna(phone):
        return ""
    phone_str = str(phone).strip()
    # Retain leading + if present, but strip other non-digit chars
    has_plus = phone_str.startswith('+')
    digits = re.sub(r'\D', '', phone_str)
    if digits:
        return f"+{digits}" if has_plus else digits
    return ""

def preprocess_patient_row(row: pd.Series) -> dict:
    """Standardizes patient record fields (Title Case names, clean emails, E.164 phones)."""
    first_name = str(row['first_name']).strip().title()
    last_name = str(row['last_name']).strip().title()
    dob = str(row['dob']).strip()
    gender = str(row['gender']).strip().capitalize() if pd.notna(row.get('gender')) else "Unknown"
    
    email = str(row['email']).strip().lower() if pd.notna(row.get('email')) else ""
    phone = clean_phone_number(row.get('phone'))

    return {
        "patient_id": str(row['patient_id']).strip(),
        "first_name": first_name,
        "last_name": last_name,
        "dob": dob,
        "gender": gender,
        "email": email,
        "phone": phone
    }

def preprocess_medical_document(file_path: Path) -> dict:
    """Cleans medical documents:

    - Extracts text content (supports txt, md, json directly; basic support for pdf).
    - Standardizes whitespace and formatting.
    - Saves clean copy to processed/documents/.
    - Parses basic metadata like Document Type, Doctor Name, Date from header fields if present.
    """
    ext = file_path.suffix.lower()
    content = ""

    if ext in {'.txt', '.md'}:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    elif ext == '.json':
        with open(file_path, 'r', encoding='utf-8') as f:
            # Load and dump with indent for clean text format
            import json
            try:
                data = json.load(f)
                content = json.dumps(data, indent=2)
            except Exception:
                f.seek(0)
                content = f.read()
    elif ext == '.pdf':
        # Fallback reading for PDF
        try:
            import pypdf
            reader = pypdf.PdfReader(file_path)
            content = "\n".join([page.extract_text() for page in reader.pages])
        except ImportError:
            # Mock text extraction if pypdf is not available
            content = f"[PDF Contents of {file_path.name} - install pypdf for extraction]"

    # Clean whitespace
    content_clean = re.sub(r'\r\n', '\n', content)
    content_clean = re.sub(r'[ \t]+', ' ', content_clean)
    content_clean = content_clean.strip()

    # Detect document type from filename or header
    doc_type = "clinical_note"
    filename_lower = file_path.name.lower()
    if "summary" in filename_lower:
        doc_type = "summary"
    elif "report" in filename_lower or "lab" in filename_lower:
        doc_type = "lab_report"
    
    # Try parsing text lines for specific headers
    lines = content_clean.split('\n')
    for line in lines[:5]:  # Look in first 5 lines
        match = re.match(r'^(document\s*type|type)\s*:\s*(.+)$', line, re.IGNORECASE)
        if match:
            doc_type = match.group(2).strip().lower().replace(' ', '_')
            break

    # Save to processed directory
    output_filename = f"{file_path.stem}_processed.txt"
    output_path = PROCESSED_DOCUMENTS_DIR / output_filename
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content_clean)

    return {
        "file_path": str(output_path),
        "filename": output_filename,
        "document_type": doc_type,
        "content": content_clean
    }
