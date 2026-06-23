"""
app/models.py — SQLAlchemy ORM Models
"""
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)

    audio_records = relationship("AudioRecord", back_populates="owner")


class AudioRecord(Base):
    __tablename__ = "audio_records_api"  # distinct from ETL pipeline's audio_records table

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
    audio_record_id = Column(Integer, ForeignKey("audio_records_api.id"), nullable=False)
    transcript = Column(String, nullable=True)
    soap_note = Column(String, nullable=True)
    summary = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    audio_record = relationship("AudioRecord", back_populates="clinical_note")
