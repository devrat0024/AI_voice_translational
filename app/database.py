"""
app/database.py — SQLAlchemy Engine & Session (FastAPI API Layer)
"""
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from app.config import DATABASE_URL

# SQLite requires check_same_thread=False for FastAPI's async context
connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """FastAPI dependency — yields a DB session and handles teardown."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
