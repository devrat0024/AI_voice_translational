import re
import logging
from pathlib import Path
import pandas as pd
from pipeline.config import init_directories
from pipeline.ingestion import scan_all_inputs, archive_file
from pipeline.validation import (
    validate_patient_records,
    validate_audio_file,
    validate_document_file,
    validate_patient_reference
)
from pipeline.preprocessing import (
    preprocess_patient_row,
    preprocess_audio,
    preprocess_medical_document
)
from pipeline.storage import (
    get_connection,
    init_db,
    insert_patient,
    insert_audio_record,
    insert_medical_document,
    get_patient_ids,
    get_stats
)

logger = logging.getLogger(__name__)

def extract_patient_id_from_filename(filename: str, existing_patient_ids: set) -> str:
    r"""Tries to extract a valid patient ID from a filename.

    Checks:
    1. If filename starts with a patient_id (e.g. 'P1001_notes.txt' -> 'P1001')
    2. Matches pattern like P\d+ or PAT-\d+ or standard alphanumeric prefixes.
    """
    # 1. Check underscore prefix
    prefix = filename.split('_')[0]
    if prefix in existing_patient_ids:
        return prefix

    # 2. Case insensitive check against existing IDs
    for p_id in existing_patient_ids:
        if filename.lower().startswith(p_id.lower()):
            return p_id

    # 3. Regex matches for P123, PAT-123, etc.
    match = re.search(r'\b(P\d+|PAT-\d+|PATIENT-\d+)\b', filename, re.IGNORECASE)
    if match:
        matched_id = match.group(1).upper()
        # Even if not in DB, return it so validation can report the referential failure
        return matched_id
        
    return prefix  # Default fallback to prefix

def run_etl_pipeline():
    """Main ETL Orchestrator function."""
    logger.info("Initializing ETL Pipeline Run...")
    init_directories()
    init_db()

    conn = get_connection()
    try:
        # Step 1: Ingest and Process Patient Demographics first
        inputs = scan_all_inputs()
        logger.info(f"Scanned inputs: {len(inputs['patients'])} patient file(s), "
                    f"{len(inputs['documents'])} document(s), "
                    f"{len(inputs['audio'])} audio file(s).")

        for patient_file in inputs['patients']:
            file_path = patient_file['path']
            logger.info(f"Ingesting patient data file: {patient_file['name']}")
            
            try:
                # Read CSV
                df = pd.read_csv(file_path)
                is_valid, errors = validate_patient_records(df)
                
                if is_valid:
                    # Preprocess and insert each row
                    for _, row in df.iterrows():
                        patient_data = preprocess_patient_row(row)
                        insert_patient(conn, patient_data)
                    conn.commit()
                    logger.info(f"Successfully processed patient file: {patient_file['name']}")
                    archive_file(file_path, success=True, subfolder="patients")
                else:
                    logger.error(f"Validation failed for patient file {patient_file['name']}: {errors}")
                    archive_file(file_path, success=False, subfolder="patients")
            except Exception as e:
                logger.error(f"Error processing patient file {patient_file['name']}: {e}")
                archive_file(file_path, success=False, subfolder="patients")

        # Refresh existing patient IDs for referential validation
        existing_patient_ids = get_patient_ids(conn)

        # Step 2: Ingest and Process Medical Documents
        for doc_file in inputs['documents']:
            file_path = doc_file['path']
            filename = doc_file['name']
            checksum = doc_file['checksum']
            
            logger.info(f"Ingesting medical document: {filename}")

            # Check if document already processed (by checksum)
            cursor = conn.cursor()
            cursor.execute("SELECT document_id, status FROM medical_documents WHERE checksum = ?", (checksum,))
            existing_doc = cursor.fetchone()
            if existing_doc:
                logger.info(f"Document {filename} with checksum {checksum[:10]} already exists in DB. Skipping.")
                archive_file(file_path, success=True, subfolder="documents")
                continue

            # Generate ID and attempt to resolve Patient ID
            doc_id = f"DOC_{checksum[:10].upper()}"
            patient_id = extract_patient_id_from_filename(filename, existing_patient_ids)

            # Validate File Integrity
            file_valid, file_err = validate_document_file(file_path)
            if not file_valid:
                logger.error(f"File validation failed for document {filename}: {file_err}")
                archive_file(file_path, success=False, subfolder="documents")
                continue

            # Validate Patient Reference
            ref_valid, ref_err = validate_patient_reference(patient_id, existing_patient_ids)
            if not ref_valid:
                logger.error(f"Patient reference check failed for document {filename}: {ref_err}")
                archive_file(file_path, success=False, subfolder="documents")
                continue

            try:
                # Preprocess and Insert
                doc_data = preprocess_medical_document(file_path)
                doc_data.update({
                    "document_id": doc_id,
                    "patient_id": patient_id,
                    "filename": filename,
                    "checksum": checksum,
                    "status": "processed"
                })
                insert_medical_document(conn, doc_data)
                conn.commit()
                logger.info(f"Successfully processed medical document {filename}")
                archive_file(file_path, success=True, subfolder="documents")
            except Exception as e:
                logger.error(f"Error processing medical document {filename}: {e}")
                archive_file(file_path, success=False, subfolder="documents")

        # Step 3: Ingest and Process Audio Files
        for audio_file in inputs['audio']:
            file_path = audio_file['path']
            filename = audio_file['name']
            checksum = audio_file['checksum']

            logger.info(f"Ingesting audio file: {filename}")

            # Check duplicate by checksum
            cursor = conn.cursor()
            cursor.execute("SELECT audio_id, status FROM audio_records WHERE checksum = ?", (checksum,))
            existing_audio = cursor.fetchone()
            if existing_audio:
                logger.info(f"Audio file {filename} with checksum {checksum[:10]} already exists in DB. Skipping.")
                archive_file(file_path, success=True, subfolder="audio")
                continue

            audio_id = f"AUD_{checksum[:10].upper()}"
            patient_id = extract_patient_id_from_filename(filename, existing_patient_ids)

            # Run validation checks
            file_valid, file_err = validate_audio_file(file_path)
            ref_valid, ref_err = validate_patient_reference(patient_id, existing_patient_ids)

            if not file_valid or not ref_valid:
                err_msg = file_err or ref_err
                logger.error(f"Validation failed for audio {filename}: {err_msg}")
                
                # Insert failed record into DB for traceability
                failed_data = {
                    "audio_id": audio_id,
                    "patient_id": patient_id if ref_valid else None,
                    "filename": filename,
                    "file_path": str(file_path),
                    "checksum": checksum,
                    "status": "failed",
                    "error_message": err_msg
                }
                try:
                    insert_audio_record(conn, failed_data)
                    conn.commit()
                except Exception as db_err:
                    logger.error(f"Failed to insert failure log into DB: {db_err}")
                
                archive_file(file_path, success=False, subfolder="audio")
                continue

            try:
                # Preprocess audio (resample, normalize, wav)
                audio_meta = preprocess_audio(file_path)
                audio_meta.update({
                    "audio_id": audio_id,
                    "patient_id": patient_id,
                    "checksum": checksum,
                    "status": "preprocessed",
                    "error_message": None
                })
                insert_audio_record(conn, audio_meta)
                conn.commit()
                logger.info(f"Successfully processed audio file {filename}")
                archive_file(file_path, success=True, subfolder="audio")
            except Exception as e:
                logger.error(f"Error processing audio file {filename}: {e}")
                
                # Log failed record
                failed_data = {
                    "audio_id": audio_id,
                    "patient_id": patient_id,
                    "filename": filename,
                    "file_path": str(file_path),
                    "checksum": checksum,
                    "status": "failed",
                    "error_message": f"Preprocessing error: {str(e)}"
                }
                try:
                    insert_audio_record(conn, failed_data)
                    conn.commit()
                except Exception:
                    pass
                archive_file(file_path, success=False, subfolder="audio")

        # Report stats
        stats = get_stats(conn)
        logger.info("ETL Pipeline Run completed successfully.")
        logger.info(f"Pipeline Stats: {stats}")
        return stats

    finally:
        conn.close()
