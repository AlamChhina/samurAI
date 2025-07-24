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

class Subscription(Base):
    __tablename__ = "subscriptions"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"))
    stripe_subscription_id = Column(String, unique=True)
    stripe_price_id = Column(String)  # monthly or yearly price ID
    plan_type = Column(String)  # "monthly" or "yearly"
    amount = Column(Float)  # Amount in USD
    currency = Column(String, default="usd")
    status = Column(String)  # active, canceled, past_due, etc.
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
    action_type = Column(String)  # "video_process", "audio_process", "transcript_process"
    job_id = Column(String, ForeignKey("jobs.id"), nullable=True)
    usage_date = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationship
    user = relationship("User", back_populates="usage_logs")

class UserPreferences(Base):
    __tablename__ = "user_preferences"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), unique=True)
    preferred_voice = Column(String, default="en-US-AriaNeural")  # Edge-TTS voice
    voice_speed = Column(String, default="1.0")  # Speech speed multiplier
    voice_pitch = Column(String, default="0Hz")  # Voice pitch adjustment
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationship
    user = relationship("User", back_populates="preferences")

class Job(Base):
    __tablename__ = "jobs"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"))
    filename = Column(String)
    source_url = Column(String)  # Original URL for jobs created from URLs
    content_hash = Column(String, index=True)  # Hash of the content for duplicate detection
    media_hash = Column(String, index=True)  # Hash of the media content only (excluding summary prompt)
    status = Column(String, default="pending")  # pending, processing, completed, failed
    transcription = Column(Text)
    transcript_file = Column(String)
    speech_file = Column(String)
    original_audio_file = Column(String)  # Original audio from video source
    summary = Column(Text)
    summary_file = Column(String)
    summary_speech_file = Column(String)
    summary_prompt = Column(Text)  # Custom prompt for summary generation
    error_message = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True))
    
    # Relationship
    user = relationship("User", back_populates="jobs")

# Helper functions for subscription management
def is_subscription_active(user: User) -> bool:
    """Check if user has an active subscription"""
    if user.subscription_tier == "free":
        return False
    
    if not user.subscription_end_date:
        return False
    
    # Ensure both datetimes are timezone-aware for comparison
    now = datetime.now(timezone.utc)
    end_date = user.subscription_end_date
    
    # If subscription_end_date is naive, assume it's UTC
    if end_date.tzinfo is None:
        end_date = end_date.replace(tzinfo=timezone.utc)
    
    return (user.subscription_status == "active" and end_date > now)

def can_process_request(user: User) -> tuple[bool, str]:
    """Check if user can process a request based on their tier and usage"""
    # Paid users have unlimited access
    if is_subscription_active(user):
        return True, "Unlimited access"
    
    # Free users have daily limits
    today = datetime.now(timezone.utc).date()
    reset_date = user.daily_usage_reset_date.date() if user.daily_usage_reset_date else today
    
    # Reset counter if it's a new day
    if reset_date < today:
        user.daily_usage_count = 0
        user.daily_usage_reset_date = datetime.now(timezone.utc)
    
    max_daily_requests = 5
    if user.daily_usage_count >= max_daily_requests:
        return False, "Daily limit of 5 requests reached. Upgrade to paid plan for unlimited access."
    
    remaining = max_daily_requests - user.daily_usage_count
    return True, f"{remaining} requests remaining today"

def increment_usage(db, user: User, action_type: str, job_id: str = None):
    """Increment user usage count and log the action"""
    # Only count usage for free users
    if not is_subscription_active(user):
        user.daily_usage_count += 1
        
        # Reset date if needed
        today = datetime.now(timezone.utc).date()
        reset_date = user.daily_usage_reset_date.date() if user.daily_usage_reset_date else today
        if reset_date < today:
            user.daily_usage_reset_date = datetime.now(timezone.utc)
    
    # Log the usage
    usage_log = UsageLog(
        user_id=user.id,
        action_type=action_type,
        job_id=job_id
    )
    db.add(usage_log)
    db.commit()

def get_content_retention_days(user: User) -> int:
    """Get the number of days content should be retained for user"""
    if is_subscription_active(user):
        return 30  # Paid users: 30 days
    else:
        return 7   # Free users: 7 days

# Create tables
Base.metadata.create_all(bind=engine)

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
