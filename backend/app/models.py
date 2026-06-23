from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Float
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from backend.app.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    role = Column(String, default="doctor", nullable=False)  # 'admin', 'doctor', 'patient'

    audio_records = relationship("AudioRecord", back_populates="owner")
    audit_logs = relationship("AuditLog", back_populates="user")
    consultation_analytics = relationship("ConsultationAnalytics", back_populates="doctor")
    doctor_analytics = relationship("DoctorAnalytics", back_populates="doctor", uselist=False)


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
    transcript = Column(String, nullable=True)  # Encrypted
    soap_note = Column(String, nullable=True)   # Encrypted
    summary = Column(String, nullable=True)     # Encrypted
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


# ── Analytics Models ──────────────────────────────────────────────────────────

class ConsultationAnalytics(Base):
    __tablename__ = "consultation_analytics"

    id = Column(Integer, primary_key=True, index=True)
    consultation_id = Column(Integer, nullable=False)
    doctor_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    patient_id = Column(Integer, nullable=True)
    audio_duration = Column(Float, default=0.0)
    processing_time = Column(Float, default=0.0)
    transcription_time = Column(Float, default=0.0)
    summary_time = Column(Float, default=0.0)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())

    doctor = relationship("User", back_populates="consultation_analytics")


class DoctorAnalytics(Base):
    __tablename__ = "doctor_analytics"

    id = Column(Integer, primary_key=True, index=True)
    doctor_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False)
    specialization = Column(String, default="General Practice")
    consultations_count = Column(Integer, default=0)
    last_login = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    active_days = Column(Integer, default=1)

    doctor = relationship("User", back_populates="doctor_analytics")


class AIEvaluation(Base):
    __tablename__ = "ai_evaluations"

    id = Column(Integer, primary_key=True, index=True)
    consultation_id = Column(Integer, nullable=False)
    wer = Column(Float, default=0.0)
    precision = Column(Float, default=1.0)
    recall = Column(Float, default=1.0)
    f1_score = Column(Float, default=1.0)
    summary_score = Column(Float, nullable=True)  # Doctor's rating 1-5


class DoctorFeedback(Base):
    __tablename__ = "doctor_feedbacks"

    id = Column(Integer, primary_key=True, index=True)
    consultation_id = Column(Integer, nullable=False)
    doctor_rating = Column(Integer, nullable=False)  # 1 to 5 stars
    feedback_text = Column(String, nullable=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
