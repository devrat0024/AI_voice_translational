import shutil
from pathlib import Path
from fastapi import UploadFile
from app.config import UPLOAD_DIR

def save_uploaded_file(upload_file: UploadFile) -> Path:
    """Saves an uploaded file to the local storage directory and returns its Path."""
    destination = UPLOAD_DIR / upload_file.filename
    with destination.open("wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
    return destination
