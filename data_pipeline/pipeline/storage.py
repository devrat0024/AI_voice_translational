import sqlite3
import logging
from datetime import datetime
from pathlib import Path
from .config import DB_PATH

logger = logging.getLogger(__name__)

def get_connection():
    """Returns a connection to the SQLite database, enabling foreign key constraints."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes the database schema (tables) if not already created."""
    conn = get_connection()
    cursor = conn.cursor()

    # Patients Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS patients (
        patient_id TEXT PRIMARY KEY,
        first_name TEXT NOT NULL,
        last_name TEXT NOT NULL,
        dob TEXT NOT NULL,
        gender TEXT,
        email TEXT,
        phone TEXT,
        created_at TEXT DEFAULT (datetime('now', 'localtime')),
        updated_at TEXT DEFAULT (datetime('now', 'localtime'))
    );
    """)

    # Audio Records Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS audio_records (
        audio_id TEXT PRIMARY KEY,
        patient_id TEXT,
        filename TEXT NOT NULL,
        file_path TEXT NOT NULL,
        file_size_bytes INTEGER,
        duration_seconds REAL,
        sample_rate INTEGER,
        channels INTEGER,
        checksum TEXT UNIQUE,
        status TEXT NOT NULL,
        error_message TEXT,
        created_at TEXT DEFAULT (datetime('now', 'localtime')),
        FOREIGN KEY (patient_id) REFERENCES patients(patient_id) ON DELETE CASCADE
    );
    """)

    # Medical Documents Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS medical_documents (
        document_id TEXT PRIMARY KEY,
        patient_id TEXT,
        document_type TEXT NOT NULL,
        filename TEXT NOT NULL,
        file_path TEXT NOT NULL,
        checksum TEXT UNIQUE,
        content TEXT,
        status TEXT NOT NULL,
        created_at TEXT DEFAULT (datetime('now', 'localtime')),
        FOREIGN KEY (patient_id) REFERENCES patients(patient_id) ON DELETE CASCADE
    );
    """)

    # Transcripts Table (Optional/Future transcripts output)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS transcripts (
        transcript_id TEXT PRIMARY KEY,
        audio_id TEXT UNIQUE,
        transcript_text TEXT NOT NULL,
        language TEXT,
        confidence REAL,
        created_at TEXT DEFAULT (datetime('now', 'localtime')),
        FOREIGN KEY (audio_id) REFERENCES audio_records(audio_id) ON DELETE CASCADE
    );
    """)

    conn.commit()
    conn.close()
    logger.info("Database schema initialized successfully.")

def insert_patient(conn, patient_data):
    """Inserts or updates a patient record."""
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO patients (patient_id, first_name, last_name, dob, gender, email, phone, updated_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now', 'localtime'))
    ON CONFLICT(patient_id) DO UPDATE SET
        first_name = excluded.first_name,
        last_name = excluded.last_name,
        dob = excluded.dob,
        gender = excluded.gender,
        email = excluded.email,
        phone = excluded.phone,
        updated_at = datetime('now', 'localtime')
    """, (
        patient_data['patient_id'],
        patient_data['first_name'],
        patient_data['last_name'],
        patient_data['dob'],
        patient_data.get('gender'),
        patient_data.get('email'),
        patient_data.get('phone')
    ))

def insert_audio_record(conn, audio_data):
    """Inserts or updates an audio record."""
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO audio_records (
        audio_id, patient_id, filename, file_path, file_size_bytes, 
        duration_seconds, sample_rate, channels, checksum, status, error_message
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(audio_id) DO UPDATE SET
        patient_id = excluded.patient_id,
        filename = excluded.filename,
        file_path = excluded.file_path,
        file_size_bytes = excluded.file_size_bytes,
        duration_seconds = excluded.duration_seconds,
        sample_rate = excluded.sample_rate,
        channels = excluded.channels,
        checksum = excluded.checksum,
        status = excluded.status,
        error_message = excluded.error_message
    """, (
        audio_data['audio_id'],
        audio_data['patient_id'],
        audio_data['filename'],
        audio_data['file_path'],
        audio_data.get('file_size_bytes'),
        audio_data.get('duration_seconds'),
        audio_data.get('sample_rate'),
        audio_data.get('channels'),
        audio_data['checksum'],
        audio_data['status'],
        audio_data.get('error_message')
    ))

def insert_medical_document(conn, doc_data):
    """Inserts or updates a medical document."""
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO medical_documents (
        document_id, patient_id, document_type, filename, file_path, checksum, content, status
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(document_id) DO UPDATE SET
        patient_id = excluded.patient_id,
        document_type = excluded.document_type,
        filename = excluded.filename,
        file_path = excluded.file_path,
        checksum = excluded.checksum,
        content = excluded.content,
        status = excluded.status
    """, (
        doc_data['document_id'],
        doc_data['patient_id'],
        doc_data['document_type'],
        doc_data['filename'],
        doc_data['file_path'],
        doc_data['checksum'],
        doc_data.get('content'),
        doc_data['status']
    ))

def get_patient_ids(conn):
    """Helper to get all patient IDs in database for verification."""
    cursor = conn.cursor()
    cursor.execute("SELECT patient_id FROM patients")
    return {row[0] for row in cursor.fetchall()}

def get_stats(conn):
    """Retrieve counts and statistics of records in the database."""
    cursor = conn.cursor()
    stats = {}
    
    cursor.execute("SELECT COUNT(*) FROM patients")
    stats['patients_count'] = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*), status FROM audio_records GROUP BY status")
    stats['audio_counts'] = {row[1]: row[0] for row in cursor.fetchall()}
    
    cursor.execute("SELECT COUNT(*), status FROM medical_documents GROUP BY status")
    stats['document_counts'] = {row[1]: row[0] for row in cursor.fetchall()}
    
    return stats
