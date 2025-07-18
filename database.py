from sqlalchemy import create_engine, Column, String, DateTime, Text, Integer, Boolean, ForeignKey, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.sql import func
import uuid
from datetime import datetime, timezone, timedelta

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
    
    # Subscription fields
    subscription_tier = Column(String, default="free")  # free, paid
    stripe_customer_id = Column(String, unique=True, nullable=True)
    subscription_status = Column(String, default="inactive")  # active, inactive, canceled, past_due
    subscription_start_date = Column(DateTime(timezone=True), nullable=True)
    subscription_end_date = Column(DateTime(timezone=True), nullable=True)
    daily_usage_count = Column(Integer, default=0)  # Reset daily for free users
    daily_usage_reset_date = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    jobs = relationship("Job", back_populates="user")
    preferences = relationship("UserPreferences", back_populates="user", uselist=False)
    subscriptions = relationship("Subscription", back_populates="user")
    usage_logs = relationship("UsageLog", back_populates="user")

class Job(Base):
    __tablename__ = "jobs"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"))
    filename = Column(String)
    content_hash = Column(String, index=True)
    status = Column(String, default="pending")
    transcription = Column(Text)
    transcript_file = Column(String)
    speech_file = Column(String)
    summary = Column(Text)
    summary_file = Column(String)
    summary_speech_file = Column(String)
    summary_prompt = Column(Text)
    error_message = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True))
    
    # Relationship
    user = relationship("User", back_populates="jobs")

class UserPreferences(Base):
    __tablename__ = "user_preferences"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), unique=True)
    preferred_voice = Column(String, default="en-US-AriaNeural")
    voice_speed = Column(String, default="1.0")
    voice_pitch = Column(String, default="0Hz")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationship
    user = relationship("User", back_populates="preferences")

class Subscription(Base):
    __tablename__ = "subscriptions"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"))
    stripe_subscription_id = Column(String, unique=True)
    stripe_price_id = Column(String)
    plan_type = Column(String)
    amount = Column(Float)
    currency = Column(String, default="usd")
    status = Column(String)
    current_period_start = Column(DateTime(timezone=True))
    current_period_end = Column(DateTime(timezone=True))
    cancel_at_period_end = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationship
    user = relationship("User", back_populates="subscriptions")

class UsageLog(Base):
    __tablename__ = "usage_logs"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"))
    action_type = Column(String)
    job_id = Column(String, ForeignKey("jobs.id"), nullable=True)
    usage_date = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationship
    user = relationship("User", back_populates="usage_logs")

# Helper functions
def is_subscription_active(user: User) -> bool:
    if user.subscription_tier == "free":
        return False
    return (user.subscription_status == "active" and 
            user.subscription_end_date and 
            user.subscription_end_date > datetime.now(timezone.utc))

def can_process_request(user: User) -> tuple[bool, str]:
    if is_subscription_active(user):
        return True, ""
    
    today = datetime.now(timezone.utc).date()
    reset_date = user.daily_usage_reset_date.date() if user.daily_usage_reset_date else today
    
    if reset_date < today:
        user.daily_usage_count = 0
        user.daily_usage_reset_date = datetime.now(timezone.utc)
    
    if user.daily_usage_count >= 5:
        return False, "Daily limit of 5 requests reached. Upgrade to paid plan for unlimited access."
    
    return True, ""

def increment_usage(db, user: User, action_type: str, job_id: str = None):
    if not is_subscription_active(user):
        user.daily_usage_count += 1
        
        today = datetime.now(timezone.utc).date()
        reset_date = user.daily_usage_reset_date.date() if user.daily_usage_reset_date else today
        if reset_date < today:
            user.daily_usage_reset_date = datetime.now(timezone.utc)
    
    usage_log = UsageLog(
        user_id=user.id,
        action_type=action_type,
        job_id=job_id
    )
    db.add(usage_log)
    db.commit()

def get_content_retention_days(user: User) -> int:
    if is_subscription_active(user):
        return 30
    else:
        return 7

# Create tables
Base.metadata.create_all(bind=engine)

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
