"""
app/pipeline/ingestion.py — Raw File Scanning & Post-processing Archival
"""
import hashlib
import shutil
import logging
from pathlib import Path
from datetime import datetime

from app.pipeline.config import (
    RAW_AUDIO_DIR,
    RAW_PATIENTS_DIR,
    RAW_DOCUMENTS_DIR,
    ARCHIVE_DIR,
    FAILED_DIR,
    SUPPORTED_AUDIO_EXTENSIONS,
    SUPPORTED_PATIENT_EXTENSIONS,
    SUPPORTED_DOCUMENT_EXTENSIONS,
)

logger = logging.getLogger(__name__)


def calculate_checksum(file_path: Path) -> str:
    """Calculates the SHA-256 checksum of a file to prevent duplicate processing."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()


def scan_raw_directory(directory_path: Path, supported_extensions: set) -> list:
    """Scans a directory for files with supported extensions.

    Returns a list of dicts: {name, path, size_bytes, checksum, extension}.
    """
    found_files = []
    if not directory_path.exists():
        logger.warning(f"Directory {directory_path} does not exist.")
        return found_files

    for file in directory_path.iterdir():
        if file.is_file() and file.suffix.lower() in supported_extensions:
            try:
                checksum = calculate_checksum(file)
                found_files.append({
                    "name": file.name,
                    "path": file,
                    "size_bytes": file.stat().st_size,
                    "checksum": checksum,
                    "extension": file.suffix.lower(),
                })
            except Exception as e:
                logger.error(f"Failed to ingest {file.name}: {e}")
    return found_files


def scan_all_inputs() -> dict:
    """Scans all raw folders for patient data, audio files, and medical documents."""
    return {
        "patients": scan_raw_directory(RAW_PATIENTS_DIR, SUPPORTED_PATIENT_EXTENSIONS),
        "audio": scan_raw_directory(RAW_AUDIO_DIR, SUPPORTED_AUDIO_EXTENSIONS),
        "documents": scan_raw_directory(RAW_DOCUMENTS_DIR, SUPPORTED_DOCUMENT_EXTENSIONS),
    }


def archive_file(file_path: Path, success: bool, subfolder: str = "") -> Path | None:
    """Moves a processed file to archive/ or failed/ with a timestamp prefix."""
    dest_root = ARCHIVE_DIR if success else FAILED_DIR
    dest_dir = dest_root / subfolder if subfolder else dest_root
    dest_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest_path = dest_dir / f"{timestamp}_{file_path.name}"
    try:
        shutil.move(str(file_path), str(dest_path))
        logger.info(f"Moved {file_path.name} → {dest_path}")
        return dest_path
    except Exception as e:
        logger.error(f"Failed to move {file_path} → {dest_path}: {e}")
        return None
