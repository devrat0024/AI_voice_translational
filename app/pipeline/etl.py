"""
app/pipeline/etl.py — ETL Pipeline Orchestrator
"""
import re
import logging
from pathlib import Path

import pandas as pd

from app.pipeline.config import init_directories
from app.pipeline.ingestion import scan_all_inputs, archive_file
from app.pipeline.validation import (
    validate_patient_records,
    validate_audio_file,
    validate_document_file,
    validate_patient_reference,
)
from app.pipeline.preprocessing import (
    preprocess_patient_row,
    preprocess_audio,
    preprocess_medical_document,
)
from app.pipeline.storage import (
    get_connection,
    init_db,
    insert_patient,
    insert_audio_record,
    insert_medical_document,
    get_patient_ids,
    get_stats,
)

logger = logging.getLogger(__name__)


def extract_patient_id_from_filename(filename: str, existing_patient_ids: set) -> str:
    r"""Extracts a valid patient ID from a filename.

    Checks:
      1. Underscore-prefix match (e.g. 'P1001_notes.txt' → 'P1001')
      2. Case-insensitive prefix match against known IDs
      3. Regex pattern match (P\d+, PAT-\d+, PATIENT-\d+)
    """
    prefix = filename.split("_")[0]
    if prefix in existing_patient_ids:
        return prefix

    for p_id in existing_patient_ids:
        if filename.lower().startswith(p_id.lower()):
            return p_id

    match = re.search(r"\b(P\d+|PAT-\d+|PATIENT-\d+)\b", filename, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    return prefix


def run_etl_pipeline() -> dict:
    """Main ETL Orchestrator — scans, validates, preprocesses, and stores all raw inputs."""
    logger.info("Initializing ETL Pipeline Run...")
    init_directories()
    init_db()

    conn = get_connection()
    try:
        inputs = scan_all_inputs()
        logger.info(
            f"Scanned inputs: {len(inputs['patients'])} patient file(s), "
            f"{len(inputs['documents'])} document(s), "
            f"{len(inputs['audio'])} audio file(s)."
        )

        # ── Step 1: Patient Demographics ──────────────────────────────────────
        for patient_file in inputs["patients"]:
            file_path = patient_file["path"]
            logger.info(f"Ingesting patient data file: {patient_file['name']}")
            try:
                df = pd.read_csv(file_path)
                is_valid, errors = validate_patient_records(df)
                if is_valid:
                    for _, row in df.iterrows():
                        insert_patient(conn, preprocess_patient_row(row))
                    conn.commit()
                    logger.info(f"Processed patient file: {patient_file['name']}")
                    archive_file(file_path, success=True, subfolder="patients")
                else:
                    logger.error(f"Validation failed for {patient_file['name']}: {errors}")
                    archive_file(file_path, success=False, subfolder="patients")
            except Exception as e:
                logger.error(f"Error processing patient file {patient_file['name']}: {e}")
                archive_file(file_path, success=False, subfolder="patients")

        existing_patient_ids = get_patient_ids(conn)

        # ── Step 2: Medical Documents ─────────────────────────────────────────
        for doc_file in inputs["documents"]:
            file_path = doc_file["path"]
            filename = doc_file["name"]
            checksum = doc_file["checksum"]
            logger.info(f"Ingesting medical document: {filename}")

            cursor = conn.cursor()
            cursor.execute(
                "SELECT document_id FROM medical_documents WHERE checksum = ?", (checksum,)
            )
            if cursor.fetchone():
                logger.info(f"Document {filename} already in DB. Skipping.")
                archive_file(file_path, success=True, subfolder="documents")
                continue

            doc_id = f"DOC_{checksum[:10].upper()}"
            patient_id = extract_patient_id_from_filename(filename, existing_patient_ids)

            file_valid, file_err = validate_document_file(file_path)
            if not file_valid:
                logger.error(f"File validation failed for {filename}: {file_err}")
                archive_file(file_path, success=False, subfolder="documents")
                continue

            ref_valid, ref_err = validate_patient_reference(patient_id, existing_patient_ids)
            if not ref_valid:
                logger.error(f"Patient reference failed for {filename}: {ref_err}")
                archive_file(file_path, success=False, subfolder="documents")
                continue

            try:
                doc_data = preprocess_medical_document(file_path)
                doc_data.update({
                    "document_id": doc_id,
                    "patient_id": patient_id,
                    "filename": filename,
                    "checksum": checksum,
                    "status": "processed",
                })
                insert_medical_document(conn, doc_data)
                conn.commit()
                logger.info(f"Processed medical document: {filename}")
                archive_file(file_path, success=True, subfolder="documents")
            except Exception as e:
                logger.error(f"Error processing document {filename}: {e}")
                archive_file(file_path, success=False, subfolder="documents")

        # ── Step 3: Audio Files ───────────────────────────────────────────────
        for audio_file in inputs["audio"]:
            file_path = audio_file["path"]
            filename = audio_file["name"]
            checksum = audio_file["checksum"]
            logger.info(f"Ingesting audio file: {filename}")

            cursor = conn.cursor()
            cursor.execute(
                "SELECT audio_id FROM audio_records WHERE checksum = ?", (checksum,)
            )
            if cursor.fetchone():
                logger.info(f"Audio {filename} already in DB. Skipping.")
                archive_file(file_path, success=True, subfolder="audio")
                continue

            audio_id = f"AUD_{checksum[:10].upper()}"
            patient_id = extract_patient_id_from_filename(filename, existing_patient_ids)

            file_valid, file_err = validate_audio_file(file_path)
            ref_valid, ref_err = validate_patient_reference(patient_id, existing_patient_ids)

            if not file_valid or not ref_valid:
                err_msg = file_err or ref_err
                logger.error(f"Validation failed for audio {filename}: {err_msg}")
                failed_data = {
                    "audio_id": audio_id,
                    "patient_id": patient_id if ref_valid else None,
                    "filename": filename,
                    "file_path": str(file_path),
                    "checksum": checksum,
                    "status": "failed",
                    "error_message": err_msg,
                }
                try:
                    insert_audio_record(conn, failed_data)
                    conn.commit()
                except Exception as db_err:
                    logger.error(f"Failed to log failure record: {db_err}")
                archive_file(file_path, success=False, subfolder="audio")
                continue

            try:
                audio_meta = preprocess_audio(file_path)
                audio_meta.update({
                    "audio_id": audio_id,
                    "patient_id": patient_id,
                    "checksum": checksum,
                    "status": "preprocessed",
                    "error_message": None,
                })
                insert_audio_record(conn, audio_meta)
                conn.commit()
                logger.info(f"Processed audio file: {filename}")
                archive_file(file_path, success=True, subfolder="audio")
            except Exception as e:
                logger.error(f"Error processing audio file {filename}: {e}")
                failed_data = {
                    "audio_id": audio_id,
                    "patient_id": patient_id,
                    "filename": filename,
                    "file_path": str(file_path),
                    "checksum": checksum,
                    "status": "failed",
                    "error_message": f"Preprocessing error: {str(e)}",
                }
                try:
                    insert_audio_record(conn, failed_data)
                    conn.commit()
                except Exception:
                    pass
                archive_file(file_path, success=False, subfolder="audio")

        stats = get_stats(conn)
        logger.info(f"ETL Pipeline Run completed. Stats: {stats}")
        return stats

    finally:
        conn.close()
