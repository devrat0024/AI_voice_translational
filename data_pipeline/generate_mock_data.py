import os
import shutil
import pandas as pd
from pathlib import Path
from data_pipeline.pipeline.config import (
    init_directories,
    RAW_PATIENTS_DIR,
    RAW_DOCUMENTS_DIR,
    RAW_AUDIO_DIR
)

def create_mock_patients():
    """Generates valid and invalid patient CSV datasets."""
    # 1. Valid patients dataset
    valid_data = {
        'patient_id': ['P1001', 'P1002', 'P1003'],
        'first_name': ['john', ' jane', 'robert '],
        'last_name': ['doe', 'smith ', ' lee'],
        'dob': ['1980-05-15', '1992-10-24', '1975-03-08'],
        'gender': ['male', 'female', 'MALE'],
        'email': ['john.doe@example.com', 'jane.smith@example.com', 'robert.lee@example.com'],
        'phone': ['+1 555-0199', '(555) 0122', '5550188']
    }
    df_valid = pd.DataFrame(valid_data)
    valid_csv_path = RAW_PATIENTS_DIR / "patients_valid.csv"
    df_valid.to_csv(valid_csv_path, index=False)
    print(f"Created valid patient CSV at: {valid_csv_path}")

    # 2. Invalid patients dataset (invalid email, invalid dob format to test validation)
    invalid_data = {
        'patient_id': ['P1004', 'P1005'],
        'first_name': ['bruce', 'clark'],
        'last_name': ['wayne', 'kent'],
        'dob': ['1939/05/27', '1938-06-18'],  # P1004 has invalid dob format (YYYY/MM/DD)
        'gender': ['male', 'male'],
        'email': ['bruce.wayne.com', 'clark.kent@dailyplanet.com'],  # P1004 has invalid email
        'phone': ['5559999', '5558888']
    }
    df_invalid = pd.DataFrame(invalid_data)
    invalid_csv_path = RAW_PATIENTS_DIR / "patients_invalid.csv"
    df_invalid.to_csv(invalid_csv_path, index=False)
    print(f"Created invalid patient CSV at: {invalid_csv_path}")

def create_mock_documents():
    """Generates mock clinical documents (text and markdown)."""
    # 1. P1001 Clinical Notes
    doc1_content = """Patient ID: P1001
Date: 2026-06-23
Document Type: clinical_note
Doctor: Dr. Sarah Connor

Patient presents with mild chest congestion and dry cough for 3 days. No fever reported.
Lungs are clear bilaterally. Prescribed rest and hydration.
"""
    doc1_path = RAW_DOCUMENTS_DIR / "P1001_clinical_notes.txt"
    with open(doc1_path, 'w', encoding='utf-8') as f:
        f.write(doc1_content)
    print(f"Created medical document at: {doc1_path}")

    # 2. P1002 Discharge Summary
    doc2_content = """# Discharge Summary
Patient ID: P1002
Date: 2026-06-23
Document Type: discharge_summary
Doctor: Dr. Stephen Strange

Patient has completed recovery protocol post minor appendectomy. Vital signs are stable.
Discharged with instructions to avoid heavy lifting.
"""
    doc2_path = RAW_DOCUMENTS_DIR / "P1002_summary.md"
    with open(doc2_path, 'w', encoding='utf-8') as f:
        f.write(doc2_content)
    print(f"Created medical document at: {doc2_path}")

    # 3. Orphan Document (P9999 doesn't exist)
    doc3_content = """Patient ID: P9999
Document Type: clinical_note
Orphan document to test patient verification errors.
"""
    doc3_path = RAW_DOCUMENTS_DIR / "P9999_orphan_report.txt"
    with open(doc3_path, 'w', encoding='utf-8') as f:
        f.write(doc3_content)
    print(f"Created medical document at: {doc3_path}")

def create_mock_audio():
    """Copies and renames existing repository MP3 files as mock patient audio sessions."""
    src_audio = Path(__file__).resolve().parent.parent / "AI_voice_translational" / "Catching Up With Friends Audio 2.mp3"
    
    if not src_audio.exists():
        print(f"Warning: Source audio file not found at {src_audio}. Skipping mock audio creation.")
        return

    # 1. Valid patient P1001 audio
    dest1 = RAW_AUDIO_DIR / "P1001_session.mp3"
    shutil.copy(src_audio, dest1)
    print(f"Copied mock audio to: {dest1}")

    # 2. Valid patient P1002 audio
    dest2 = RAW_AUDIO_DIR / "P1002_recording.mp3"
    shutil.copy(src_audio, dest2)
    print(f"Copied mock audio to: {dest2}")

    # 3. Orphan patient P8888 audio (referential integrity failure test)
    dest3 = RAW_AUDIO_DIR / "P8888_orphan_session.mp3"
    shutil.copy(src_audio, dest3)
    print(f"Copied mock audio to: {dest3}")

def main():
    print("Initializing directories...")
    init_directories()
    
    print("Generating mock patient records...")
    create_mock_patients()
    
    print("Generating mock medical documents...")
    create_mock_documents()
    
    print("Generating mock patient audios...")
    create_mock_audio()
    
    print("\nMock data generation complete!")

if __name__ == "__main__":
    main()
