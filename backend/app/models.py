from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from backend.app.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    # Role-Based Access Control: roles are 'admin', 'doctor', 'patient'
    role = Column(String, default="doctor", nullable=False)

    audio_records = relationship("AudioRecord", back_populates="owner")
    audit_logs = relationship("AuditLog", back_populates="user")


class AudioRecord(Base):
    __tablename__ = "audio_records"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    filepath = Column(String, nullable=False)
    uploaded_at = Column(DateTime(timezone=True), server_default=func.now())
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    owner = relationship("User", back_populates="audio_records")
    clinical_note = relationship("ClinicalNote", back_populates="audio_record", uselist=False)


class ClinicalNote(Base):
    __tablename__ = "clinical_notes"

    id = Column(Integer, primary_key=True, index=True)
    audio_record_id = Column(Integer, ForeignKey("audio_records.id"), nullable=False)
    transcript = Column(String, nullable=True)  # Will be encrypted
    soap_note = Column(String, nullable=True)   # Will be encrypted
    summary = Column(String, nullable=True)     # Will be encrypted
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    audio_record = relationship("AudioRecord", back_populates="clinical_note")


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    action = Column(String, nullable=False)
    resource = Column(String, nullable=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    details = Column(String, nullable=True)

    user = relationship("User", back_populates="audit_logs")
