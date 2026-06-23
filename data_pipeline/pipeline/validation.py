import re
import pandas as pd
import logging
from pathlib import Path
from pydub import AudioSegment
from pipeline.config import SUPPORTED_AUDIO_EXTENSIONS, SUPPORTED_DOCUMENT_EXTENSIONS

logger = logging.getLogger(__name__)

def validate_patient_records(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """Validates patient demographic records in a Pandas DataFrame.

    Checks schema, missing values, date format, and email format.
    Returns (is_valid, errors_list).
    """
    errors = []
    required_cols = {'patient_id', 'first_name', 'last_name', 'dob'}
    missing_cols = required_cols - set(df.columns)
    
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")
        return False, errors

    # Check for empty patient_ids
    if df['patient_id'].isna().any() or (df['patient_id'] == '').any():
        errors.append("patient_id column contains null or empty values.")

    # Validate Date of Birth (dob) format: YYYY-MM-DD
    dob_pattern = re.compile(r'^\d{4}-\d{2}-\d{2}$')
    for idx, row in df.iterrows():
        p_id = row.get('patient_id', f"Row_{idx}")
        dob = str(row['dob']).strip()
        if not dob_pattern.match(dob):
            errors.append(f"Patient {p_id} has invalid dob '{dob}' (expected YYYY-MM-DD).")
        else:
            try:
                pd.to_datetime(dob)
            except Exception:
                errors.append(f"Patient {p_id} has unparseable dob '{dob}'.")

        # Validate email if present
        email = row.get('email')
        if pd.notna(email) and str(email).strip():
            email = str(email).strip()
            if '@' not in email or '.' not in email.split('@')[-1]:
                errors.append(f"Patient {p_id} has invalid email address '{email}'.")

    is_valid = len(errors) == 0
    return is_valid, errors

def validate_audio_file(file_path: Path) -> tuple[bool, str]:
    """Validates if an audio file is not corrupted and matches supported formats.

    Uses pydub to attempt to open the file.
    Returns (is_valid, error_message).
    """
    if file_path.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
        return False, f"Unsupported audio extension: {file_path.suffix}"
    
    if file_path.stat().st_size == 0:
        return False, "Audio file is empty (0 bytes)."

    try:
        # Try loading metadata or a small chunk of audio using pydub
        # If static_ffmpeg is required, pydub will resolve ffmpeg path automatically
        # after importing static_ffmpeg in Python.
        import static_ffmpeg
        static_ffmpeg.add_paths()
        
        audio = AudioSegment.from_file(file_path)
        # Check basic characteristics
        if len(audio) == 0:
            return False, "Audio file contains no audio data."
        return True, ""
    except Exception as e:
        logger.error(f"Validation failed for audio {file_path.name}: {e}")
        return False, f"Corrupted file or format decoding error: {str(e)}"

def validate_document_file(file_path: Path) -> tuple[bool, str]:
    """Validates medical document file extension and contents."""
    if file_path.suffix.lower() not in SUPPORTED_DOCUMENT_EXTENSIONS:
        return False, f"Unsupported document extension: {file_path.suffix}"

    if file_path.stat().st_size == 0:
        return False, "Document file is empty (0 bytes)."

    try:
        if file_path.suffix.lower() in {'.txt', '.md', '.json'}:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if not content.strip():
                    return False, "Document contains only whitespace."
        elif file_path.suffix.lower() == '.pdf':
            # Basic PDF sanity check - first few bytes must be %PDF
            with open(file_path, 'rb') as f:
                header = f.read(4)
                if header != b'%PDF':
                    return False, "Invalid PDF header/file signature."
        return True, ""
    except Exception as e:
        return False, f"Document reading error: {str(e)}"

def validate_patient_reference(patient_id: str, existing_patient_ids: set) -> tuple[bool, str]:
    """Checks if the referenced patient_id exists in the database.

    This prevents orphaned records.
    """
    if not patient_id:
        return False, "Missing patient_id."
    if patient_id not in existing_patient_ids:
        return False, f"Referenced patient_id '{patient_id}' does not exist in the database."
    return True, ""
