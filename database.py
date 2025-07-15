from sqlalchemy import create_engine, Column, String, DateTime, Text, Integer, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.sql import func
import uuid

DATABASE_URL = "sqlite:///./video_audio_service.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    google_id = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    name = Column(String)
    picture = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationship
    jobs = relationship("Job", back_populates="user")

class Job(Base):
    __tablename__ = "jobs"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"))
    filename = Column(String)
    status = Column(String, default="pending")  # pending, processing, completed, failed
    transcription = Column(Text)
    transcript_file = Column(String)
    speech_file = Column(String)
    summary = Column(Text)
    summary_file = Column(String)
    summary_speech_file = Column(String)
    error_message = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True))
    
    # Relationship
    user = relationship("User", back_populates="jobs")

# Create tables
Base.metadata.create_all(bind=engine)

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
