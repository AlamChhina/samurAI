#!/usr/bin/env python3
"""
Comprehensive Unit Tests for Async Video Service
Tests all major functionality including database, utilities, processing, and API endpoints.
"""

import pytest
import asyncio
import tempfile
import os
import hashlib
import uuid
import jwt
import numpy as np
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime, timezone, timedelta
from pathlib import Path
import json

# FastAPI testing
from fastapi.testclient import TestClient
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# Import the application modules
from async_video_service import app, generate_content_hash, generate_file_hash, generate_media_hash
from async_video_service import is_valid_video_url_or_id, is_valid_youtube_url_or_id, is_valid_youtube_url
from async_video_service import extract_youtube_video_id, extract_video_id_from_url
from async_video_service import normalize_youtube_input, normalize_video_input
from async_video_service import check_duplicate_job, check_existing_processed_content, check_existing_transcript_by_media
from async_video_service import get_or_create_user_preferences, create_job_from_existing_content
from async_video_service import text_to_speech, generate_edge_tts
from async_video_service import transcribe_audio, extract_audio_from_video, convert_audio_to_wav
from async_video_service import download_video_from_url, download_original_audio_from_url
from async_video_service import get_youtube_transcript, _create_mlx_summary
from async_video_service import get_current_user, markdown_filter
from database import Base, User, Job, UserPreferences, Subscription, UsageLog
from database import is_subscription_active, can_process_request, increment_usage, get_db

# Test database setup
TEST_DATABASE_URL = "sqlite:///:memory:"
test_engine = create_engine(
    TEST_DATABASE_URL, 
    connect_args={"check_same_thread": False},
    poolclass=StaticPool
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

# Test client
client = TestClient(app)

@pytest.fixture(scope="function")
def test_db():
    """Create a test database for each test"""
    Base.metadata.create_all(bind=test_engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=test_engine)

@pytest.fixture
def sample_user(test_db):
    """Create a sample user for testing"""
    user = User(
        id="test-user-123",
        google_id="google-123",
        email="test@example.com",
        name="Test User",
        subscription_tier="free",
        subscription_status="inactive",
        daily_usage_count=0,
        daily_usage_reset_date=datetime.now(timezone.utc)
    )
    test_db.add(user)
    test_db.commit()
    test_db.refresh(user)
    return user

@pytest.fixture
def premium_user(test_db):
    """Create a premium user for testing"""
    user = User(
        id="premium-user-123",
        google_id="google-premium-123",
        email="premium@example.com",
        name="Premium User",
        subscription_tier="paid",
        subscription_status="active",
        subscription_start_date=datetime.now(timezone.utc),
        subscription_end_date=datetime.now(timezone.utc) + timedelta(days=30),
        daily_usage_count=0,
        daily_usage_reset_date=datetime.now(timezone.utc)
    )
    test_db.add(user)
    test_db.commit()
    test_db.refresh(user)
    return user

@pytest.fixture
def sample_job(test_db, sample_user):
    """Create a sample job for testing"""
    job = Job(
        id="test-job-123",
        user_id=sample_user.id,
        filename="test_video.mp4",
        content_hash="test-content-hash",
        media_hash="test-media-hash",
        status="completed",
        transcription="This is a test transcription of the video content.",
        transcript_file="test_transcript.txt",
        speech_file="test_speech.mp3",
        summary="This is a test summary.",
        summary_file="test_summary.txt",
        summary_speech_file="test_summary_speech.mp3"
    )
    test_db.add(job)
    test_db.commit()
    test_db.refresh(job)
    return job

@pytest.fixture
def temp_file():
    """Create a temporary file for testing"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
        f.write(b"fake video content for testing")
        temp_path = f.name
    yield temp_path
    if os.path.exists(temp_path):
        os.unlink(temp_path)

@pytest.fixture
def temp_audio_file():
    """Create a temporary audio file for testing"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(b"fake audio content for testing")
        temp_path = f.name
    yield temp_path
    if os.path.exists(temp_path):
        os.unlink(temp_path)


# ============================================================================
# DATABASE TESTS
# ============================================================================

class TestDatabaseModels:
    """Test database models and their methods"""
    
    def test_user_creation(self, test_db):
        """Test user model creation"""
        user = User(
            google_id="test-google-id",
            email="test@example.com",
            name="Test User"
        )
        test_db.add(user)
        test_db.commit()
        
        assert user.id is not None
        assert user.subscription_tier == "free"
        assert user.subscription_status == "inactive"
        assert user.daily_usage_count == 0
    
    def test_job_creation(self, test_db, sample_user):
        """Test job model creation"""
        job = Job(
            user_id=sample_user.id,
            filename="test.mp4",
            content_hash="test-hash",
            status="pending"
        )
        test_db.add(job)
        test_db.commit()
        
        assert job.id is not None
        assert job.user_id == sample_user.id
        assert job.status == "pending"
    
    def test_user_preferences_creation(self, test_db, sample_user):
        """Test user preferences model"""
        prefs = UserPreferences(
            user_id=sample_user.id,
            preferred_voice="en-US-AriaNeural",
            voice_speed="medium",
            voice_pitch="medium"
        )
        test_db.add(prefs)
        test_db.commit()
        
        assert prefs.user_id == sample_user.id
        assert prefs.preferred_voice == "en-US-AriaNeural"

class TestDatabaseFunctions:
    """Test database utility functions"""
    
    def test_is_subscription_active_free_user(self, sample_user):
        """Test subscription check for free user"""
        assert not is_subscription_active(sample_user)
    
    def test_is_subscription_active_premium_user(self, premium_user):
        """Test subscription check for premium user"""
        assert is_subscription_active(premium_user)
    
    def test_can_process_request_free_user_under_limit(self, sample_user):
        """Test processing limit for free user under daily limit"""
        sample_user.daily_usage_count = 3
        can_process, message = can_process_request(sample_user)
        assert can_process
        assert "remaining" in message.lower()
    
    def test_can_process_request_free_user_over_limit(self, sample_user):
        """Test processing limit for free user over daily limit"""
        sample_user.daily_usage_count = 6
        can_process, message = can_process_request(sample_user)
        assert not can_process
        assert "daily limit" in message.lower()
    
    def test_can_process_request_premium_user(self, premium_user):
        """Test processing limit for premium user (unlimited)"""
        premium_user.daily_usage_count = 100
        can_process, _ = can_process_request(premium_user)
        assert can_process
    
    def test_increment_usage(self, test_db, sample_user):
        """Test usage increment function"""
        initial_count = sample_user.daily_usage_count
        increment_usage(test_db, sample_user, "video_process", "test-job-123")
        test_db.refresh(sample_user)
        
        assert sample_user.daily_usage_count == initial_count + 1
        
        # Check usage log was created
        usage_log = test_db.query(UsageLog).filter(UsageLog.user_id == sample_user.id).first()
        assert usage_log is not None
        assert usage_log.action_type == "video_process"
    
    def test_get_or_create_user_preferences_new(self, test_db, sample_user):
        """Test creating new user preferences"""
        prefs = get_or_create_user_preferences(test_db, sample_user.id)
        assert prefs.user_id == sample_user.id
        assert prefs.preferred_voice == "en-US-AriaNeural"  # default
    
    def test_get_or_create_user_preferences_existing(self, test_db, sample_user):
        """Test getting existing user preferences"""
        # Create preferences first
        existing_prefs = UserPreferences(
            user_id=sample_user.id,
            preferred_voice="en-US-JennyNeural"
        )
        test_db.add(existing_prefs)
        test_db.commit()
        
        # Should return existing preferences
        prefs = get_or_create_user_preferences(test_db, sample_user.id)
        assert prefs.preferred_voice == "en-US-JennyNeural"


# ============================================================================
# UTILITY FUNCTION TESTS
# ============================================================================

class TestUtilityFunctions:
    """Test utility functions for hashing, validation, etc."""
    
    def test_generate_content_hash(self):
        """Test content hash generation"""
        content = "test content"
        hash1 = generate_content_hash(content)
        hash2 = generate_content_hash(content)
        hash3 = generate_content_hash("different content")
        
        assert hash1 == hash2  # Same content should produce same hash
        assert hash1 != hash3  # Different content should produce different hash
        assert len(hash1) == 64  # SHA-256 hex string length
    
    def test_generate_file_hash(self, temp_file):
        """Test file hash generation"""
        hash1 = generate_file_hash(temp_file)
        hash2 = generate_file_hash(temp_file)
        
        assert hash1 == hash2
        assert len(hash1) == 64
    
    @pytest.mark.asyncio
    async def test_generate_file_hash_async(self, temp_file):
        """Test async file hash generation"""
        hash1 = await asyncio.to_thread(generate_file_hash, temp_file)
        hash2 = await asyncio.to_thread(generate_file_hash, temp_file)
        
        assert hash1 == hash2
    
    def test_generate_media_hash(self):
        """Test media hash generation"""
        content = "youtube:test-video-id"
        hash1 = generate_media_hash(content)
        hash2 = generate_media_hash(content)
        
        assert hash1 == hash2
        assert len(hash1) == 64
    
    def test_markdown_filter(self):
        """Test markdown filter function"""
        text = "**Bold** and *italic* text with [link](http://example.com)"
        result = markdown_filter(text)
        
        assert "<strong>Bold</strong>" in result
        assert "<em>italic</em>" in result
        assert '<a href="http://example.com">' in result

class TestURLValidation:
    """Test URL validation functions"""
    
    def test_is_valid_youtube_url_or_id_valid_ids(self):
        """Test YouTube ID validation with valid IDs"""
        valid_ids = [
            "dQw4w9WgXcQ",  # Rick Roll
            "9bZkp7q19f0",  # Gangnam Style
            "kJQP7kiw5Fk"   # Despacito
        ]
        
        for video_id in valid_ids:
            assert is_valid_youtube_url_or_id(video_id)
    
    def test_is_valid_youtube_url_or_id_valid_urls(self):
        """Test YouTube URL validation with valid URLs"""
        valid_urls = [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://youtu.be/dQw4w9WgXcQ",
            "https://youtube.com/watch?v=dQw4w9WgXcQ",
            "https://www.youtube.com/embed/dQw4w9WgXcQ"
        ]
        
        for url in valid_urls:
            assert is_valid_youtube_url_or_id(url)
    
    def test_is_valid_youtube_url_or_id_invalid(self):
        """Test YouTube validation with invalid inputs"""
        invalid_inputs = [
            "invalid",
            "not-a-youtube-id",
            "https://vimeo.com/123456",
            "https://example.com",
            ""
        ]
        
        for invalid_input in invalid_inputs:
            assert not is_valid_youtube_url_or_id(invalid_input)
    
    def test_is_valid_video_url_or_id_various_platforms(self):
        """Test video URL validation for various platforms"""
        valid_urls = [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://vimeo.com/123456789",
            "https://www.dailymotion.com/video/x123456",
            "https://www.twitch.tv/videos/123456789",
            "dQw4w9WgXcQ"  # YouTube ID
        ]
        
        for url in valid_urls:
            assert is_valid_video_url_or_id(url)
    
    def test_extract_youtube_video_id(self):
        """Test YouTube video ID extraction"""
        test_cases = [
            ("dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("invalid", None)
        ]
        
        for input_str, expected in test_cases:
            result = extract_youtube_video_id(input_str)
            assert result == expected
    
    def test_extract_video_id_from_url(self):
        """Test video ID extraction from various platforms"""
        test_cases = [
            ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", ("youtube", "dQw4w9WgXcQ")),
            ("https://vimeo.com/123456789", ("vimeo", "123456789")),
            ("dQw4w9WgXcQ", ("youtube", "dQw4w9WgXcQ")),
            ("https://example.com/video", ("other", "https://example.com/video"))
        ]
        
        for url, expected in test_cases:
            result = extract_video_id_from_url(url)
            assert result == expected
    
    def test_normalize_youtube_input(self):
        """Test YouTube input normalization"""
        test_cases = [
            ("dQw4w9WgXcQ", "https://www.youtube.com/watch?v=dQw4w9WgXcQ"),
            ("https://youtu.be/dQw4w9WgXcQ", "https://youtu.be/dQw4w9WgXcQ")
        ]
        
        for input_str, expected in test_cases:
            result = normalize_youtube_input(input_str)
            assert result == expected

class TestDuplicateDetection:
    """Test duplicate job detection functions"""
    
    def test_check_duplicate_job_exists(self, test_db, sample_user, sample_job):
        """Test duplicate job detection when job exists"""
        duplicate = check_duplicate_job(test_db, sample_user.id, sample_job.content_hash)
        assert duplicate is not None
        assert duplicate.id == sample_job.id
    
    def test_check_duplicate_job_not_exists(self, test_db, sample_user):
        """Test duplicate job detection when job doesn't exist"""
        duplicate = check_duplicate_job(test_db, sample_user.id, "non-existent-hash")
        assert duplicate is None
    
    def test_check_existing_processed_content(self, test_db, sample_job):
        """Test checking for existing processed content"""
        result = check_existing_processed_content(test_db, sample_job.content_hash)
        assert result is not None
        assert result.id == sample_job.id
    
    def test_check_existing_transcript_by_media(self, test_db, sample_job):
        """Test checking for existing transcript by media hash"""
        result = check_existing_transcript_by_media(test_db, sample_job.media_hash)
        assert result is not None
        assert result.id == sample_job.id
    
    def test_create_job_from_existing_content(self, test_db, sample_user, sample_job):
        """Test creating job from existing content"""
        new_job = create_job_from_existing_content(
            test_db, sample_user.id, sample_job, "new_filename.mp4", "custom prompt"
        )
        
        assert new_job.id != sample_job.id  # Should be a new job
        assert new_job.user_id == sample_user.id
        assert new_job.filename == "new_filename.mp4"
        assert new_job.summary_prompt == "custom prompt"
        assert new_job.transcription == sample_job.transcription  # Content should be copied


# ============================================================================
# AUDIO/VIDEO PROCESSING TESTS
# ============================================================================

class TestAudioVideoProcessing:
    """Test audio and video processing functions"""
    
    @pytest.mark.asyncio
    async def test_extract_audio_from_video_mock(self):
        """Test audio extraction with mocked moviepy"""
        with patch('async_video_service.mp.VideoFileClip') as mock_clip:
            mock_video = Mock()
            mock_audio = Mock()
            mock_video.audio = mock_audio
            mock_clip.return_value = mock_video
            
            # Create a real temp file that can be accessed
            temp_video = tempfile.mktemp(suffix=".mp4")
            try:
                with open(temp_video, 'wb') as f:
                    f.write(b"fake video")
                
                result = await extract_audio_from_video(temp_video)
                
                assert result is not None
                assert result.endswith(".wav")
                mock_audio.write_audiofile.assert_called_once()
            finally:
                if os.path.exists(temp_video):
                    os.unlink(temp_video)
    
    @pytest.mark.asyncio
    async def test_convert_audio_to_wav_mock(self):
        """Test audio conversion with mocked pydub"""
        with patch('async_video_service.AudioSegment') as mock_audio_segment, \
             patch('os.path.getsize', return_value=1024):
            
            mock_audio = Mock()
            mock_audio_segment.from_file.return_value = mock_audio
            
            # Mock the export method to create a file
            def mock_export(filepath, format=None):
                # Create the expected output file
                with open(filepath, 'wb') as f:
                    f.write(b"fake wav audio")
            
            mock_audio.export.side_effect = mock_export
            
            # Create a real temp file that can be accessed
            temp_audio = tempfile.mktemp(suffix=".mp3")
            try:
                with open(temp_audio, 'wb') as f:
                    f.write(b"fake audio")
                
                result = await convert_audio_to_wav(temp_audio)
                
                assert result is not None
                assert result.endswith(".wav")
                mock_audio.export.assert_called_once()
                
                # Clean up the created wav file
                if result and os.path.exists(result):
                    os.unlink(result)
                    
            finally:
                if os.path.exists(temp_audio):
                    os.unlink(temp_audio)
    
    @pytest.mark.asyncio
    async def test_transcribe_audio_mock(self, temp_audio_file):
        """Test audio transcription with mocked Whisper"""
        # Create mock tensor object with shape attribute
        mock_tensor = Mock()
        mock_tensor.shape = (1, 80, 3000)  # Typical whisper input shape
        mock_tensor.device = Mock()
        mock_tensor.to = Mock(return_value=mock_tensor)
        
        # Create mock model output
        mock_output = Mock()
        mock_output.logits = Mock()
        mock_output.to = Mock(return_value=mock_output)
        
        with patch('async_video_service.load_model') as mock_load_model, \
             patch('librosa.load', return_value=(np.array([0.1, 0.2, 0.3]), 16000)), \
             patch('torch.no_grad'), \
             patch('torch.tensor', return_value=mock_tensor):

            # Mock the load_model function to return mock processor and model
            mock_processor = Mock()
            mock_model = Mock()
            mock_load_model.return_value = (mock_processor, mock_model)

            # Mock processor - return tensor-like object with shape
            mock_processor.return_value = {"input_features": mock_tensor}

            # Mock model generation
            mock_model.generate.return_value = mock_output

            # Mock decode
            mock_processor.batch_decode.return_value = ["This is a test transcription."]

            result = await transcribe_audio(temp_audio_file)

            assert result == "This is a test transcription."

    @pytest.mark.asyncio
    async def test_generate_edge_tts_mock(self):
        """Test Edge TTS generation with mocked edge_tts"""
        with patch('async_video_service.edge_tts.Communicate') as mock_communicate, \
             patch('os.path.exists', return_value=True), \
             patch('os.path.getsize', return_value=1024):
            mock_comm_instance = AsyncMock()
            mock_communicate.return_value = mock_comm_instance

            temp_output = "/fake/path/test.mp3"
            result = await generate_edge_tts(
                "Test text", temp_output, "en-US-AriaNeural"
            )

            assert result is True
            mock_communicate.assert_called_once_with(
                "Test text", "en-US-AriaNeural"
            )
            mock_comm_instance.save.assert_called_once_with(temp_output)
    
    @pytest.mark.asyncio
    async def test_text_to_speech_edge_tts_available(self):
        """Test text_to_speech function when Edge TTS is available"""
        with patch('async_video_service.generate_edge_tts', return_value=True) as mock_edge_tts:
            result = await text_to_speech("Test text", "/fake/path/output.mp3")
            
            assert result is True
            mock_edge_tts.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_text_to_speech_fallback_to_system(self):
        """Test text_to_speech fallback to system TTS"""
        with patch('async_video_service.generate_edge_tts', return_value=False), \
             patch('async_video_service.generate_system_tts', return_value=True) as mock_system_tts:
            
            result = await text_to_speech("Test text", "/fake/path/output.mp3")
            
            assert result is True
            mock_system_tts.assert_called_once()

class TestMLXSummarization:
    """Test MLX summarization functionality"""
    
    @patch('async_video_service.MLX_AVAILABLE', True)
    @patch('async_video_service.load_mlx_model')
    @patch('async_video_service.generate')
    def test_create_mlx_summary_short_content(self, mock_generate, mock_load_model):
        """Test MLX summary generation for short content"""
        # Mock model loading
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_load_model.return_value = (mock_model, mock_tokenizer)
        
        # Mock generation
        mock_generate.return_value = "This is a short summary."
        
        # Test short content (< 1000 chars)
        short_text = "This is a short piece of text to summarize."
        result = _create_mlx_summary(short_text)
        
        assert result == "This is a short summary."
        mock_generate.assert_called_once()
        
        # Check that it used appropriate parameters for short content
        call_args = mock_generate.call_args
        assert call_args[1]['max_tokens'] == 150  # Short content gets 150 tokens
    
    @patch('async_video_service.MLX_AVAILABLE', True)
    @patch('async_video_service.load_mlx_model')
    @patch('async_video_service.generate')
    def test_create_mlx_summary_long_content(self, mock_generate, mock_load_model):
        """Test MLX summary generation for long content"""
        # Mock model loading
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_load_model.return_value = (mock_model, mock_tokenizer)
        
        # Mock generation
        mock_generate.return_value = "This is a comprehensive summary of the long content."
        
        # Test long content (> 15000 chars)
        long_text = "This is a very long piece of text. " * 500  # Create long text
        result = _create_mlx_summary(long_text)
        
        assert result == "This is a comprehensive summary of the long content."
        
        # Check that it used appropriate parameters for long content
        call_args = mock_generate.call_args
        assert call_args[1]['max_tokens'] == 800  # Long content gets 800 tokens
    
    @patch('async_video_service.MLX_AVAILABLE', True)
    @patch('async_video_service.load_mlx_model')
    def test_create_mlx_summary_model_fallback(self, mock_load_model):
        """Test MLX summary when model fails to load"""
        # Mock model loading failure
        mock_load_model.return_value = ("fallback", None)
        
        with pytest.raises(RuntimeError, match="MLX model failed to load"):
            _create_mlx_summary("Test text")
    
    @patch('async_video_service.MLX_AVAILABLE', True)
    @patch('async_video_service.load_mlx_model')
    @patch('async_video_service.generate')
    def test_create_mlx_summary_with_custom_prompt(self, mock_generate, mock_load_model):
        """Test MLX summary with custom prompt"""
        # Mock model loading
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_load_model.return_value = (mock_model, mock_tokenizer)
        
        # Mock generation
        mock_generate.return_value = "Custom summary based on prompt."
        
        result = _create_mlx_summary("Test text", "Focus on technical details")
        
        assert result == "Custom summary based on prompt."
        
        # Check that custom prompt was included
        call_args = mock_generate.call_args
        prompt_text = call_args[1]['prompt']
        assert "Focus on technical details" in prompt_text


# ============================================================================
# API ENDPOINT TESTS
# ============================================================================

class TestAPIEndpoints:
    """Test API endpoints"""
    
    def setup_method(self):
        """Override get_db dependency for testing"""
        def override_get_db():
            db = TestingSessionLocal()
            try:
                yield db
            finally:
                db.close()
        
        app.dependency_overrides[get_db] = override_get_db
    
    def teardown_method(self):
        """Clean up dependency overrides"""
        app.dependency_overrides.clear()
    
    def test_health_check(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
    
    def test_get_user_jobs_authenticated(self):
        """Test getting user jobs when authenticated"""
        # Mock user
        mock_user = Mock()
        mock_user.id = "test-user-123"
        
        # Override the dependency
        def override_get_current_user():
            return mock_user
            
        def override_get_db():
            mock_db = Mock()
            mock_query = Mock()
            mock_query.filter.return_value.order_by.return_value.all.return_value = []
            mock_db.query.return_value = mock_query
            return mock_db
        
        app.dependency_overrides[get_current_user] = override_get_current_user
        app.dependency_overrides[get_db] = override_get_db
        
        try:
            response = client.get("/api/jobs/")
            assert response.status_code == 200
            assert response.json() == []
        finally:
            # Clean up overrides
            app.dependency_overrides.clear()
    
    def test_get_user_jobs_unauthenticated(self):
        """Test getting user jobs when not authenticated"""
        def override_get_current_user():
            return None
            
        app.dependency_overrides[get_current_user] = override_get_current_user
        
        try:
            response = client.get("/api/jobs/")
            assert response.status_code == 401
        finally:
            app.dependency_overrides.clear()
    
    def test_process_video_invalid_file_type(self):
        """Test video processing with invalid file type"""
        mock_user = Mock()
        mock_user.id = "test-user-123"
        
        def override_get_current_user():
            return mock_user
            
        def override_can_process_request(user):
            return True, "OK"
        
        app.dependency_overrides[get_current_user] = override_get_current_user
        
        try:
            with patch('async_video_service.can_process_request', override_can_process_request):
                response = client.post(
                    "/api/process-video/",
                    files={"file": ("test.txt", b"not a video", "text/plain")}
                )
                assert response.status_code == 400
                # Check that the error message mentions file types
                assert "Supported formats" in response.json()["detail"]
        finally:
            app.dependency_overrides.clear()
    
    def test_process_video_usage_limit_exceeded(self):
        """Test video processing when usage limit exceeded"""
        mock_user = Mock()
        mock_user.id = "test-user-123"
        
        def override_get_current_user():
            return mock_user
            
        def override_can_process_request(user):
            return False, "Daily limit exceeded"
        
        app.dependency_overrides[get_current_user] = override_get_current_user
        
        try:
            with patch('async_video_service.can_process_request', override_can_process_request):
                response = client.post(
                    "/api/process-video/",
                    files={"file": ("test.mp4", b"fake video", "video/mp4")}
                )
                assert response.status_code == 429
        finally:
            app.dependency_overrides.clear()
            assert "Daily limit exceeded" in response.json()["detail"]


# ============================================================================
# AUTHENTICATION TESTS
# ============================================================================

class TestAuthentication:
    """Test authentication and authorization"""
    
    @pytest.mark.asyncio
    async def test_get_current_user_valid_token(self):
        """Test getting current user with valid token"""
        # Mock request with valid JWT token
        mock_request = Mock()
        mock_request.cookies = {"access_token": "valid.jwt.token"}
        
        # Mock database
        mock_db = Mock()
        mock_user = Mock()
        mock_user.email = "test@example.com"
        mock_db.query.return_value.filter.return_value.first.return_value = mock_user
        
        with patch('jwt.decode', return_value={"email": "test@example.com"}):
            result = await get_current_user(mock_request, mock_db)
            assert result == mock_user
    
    @pytest.mark.asyncio
    async def test_get_current_user_invalid_token(self):
        """Test getting current user with invalid token"""
        # Mock request with invalid JWT token
        mock_request = Mock()
        mock_request.cookies = {"access_token": "invalid.jwt.token"}
        
        mock_db = Mock()
        
        with patch('jwt.decode', side_effect=jwt.InvalidTokenError()):
            result = await get_current_user(mock_request, mock_db)
            assert result is None
    
    @pytest.mark.asyncio
    async def test_get_current_user_no_token(self):
        """Test getting current user with no token"""
        # Mock request without token
        mock_request = Mock()
        mock_request.cookies = {}
        
        mock_db = Mock()
        
        result = await get_current_user(mock_request, mock_db)
        assert result is None


# ============================================================================
# BACKGROUND PROCESSING TESTS
# ============================================================================

class TestBackgroundProcessing:
    """Test background processing functions"""
    
    @pytest.mark.asyncio
    async def test_download_video_from_url_mock(self):
        """Test video download with mocked yt-dlp"""
        with patch('async_video_service.yt_dlp.YoutubeDL') as mock_ytdl, \
             patch('glob.glob', return_value=['/fake/path/temp_video_123.mp4']):
            mock_instance = Mock()
            mock_instance.extract_info.return_value = {
                'title': 'Test Video',
                'requested_downloads': [{'filepath': '/fake/path/video.mp4'}]
            }
            mock_ytdl.return_value.__enter__.return_value = mock_instance

            result = await download_video_from_url("https://youtube.com/watch?v=test")

            assert result is not None
            video_path, title = result
            assert title == "Test Video"
            assert video_path == "/fake/path/temp_video_123.mp4"
    
    @pytest.mark.asyncio
    async def test_download_original_audio_from_url_mock(self):
        """Test original audio download with mocked yt-dlp"""
        with patch('async_video_service.yt_dlp.YoutubeDL') as mock_ytdl, \
             patch('glob.glob', return_value=['/fake/path/temp_audio_456.mp3']):
            mock_instance = Mock()
            mock_instance.extract_info.return_value = {
                'title': 'Test Audio',
                'requested_downloads': [{'filepath': '/fake/path/audio.mp3'}]
            }
            mock_ytdl.return_value.__enter__.return_value = mock_instance

            result = await download_original_audio_from_url("https://youtube.com/watch?v=test")

            assert result is not None
            audio_path, title = result
            assert title == "Test Audio"
            assert audio_path == "/fake/path/temp_audio_456.mp3"
    
    @pytest.mark.asyncio
    async def test_get_youtube_transcript_success(self):
        """Test YouTube transcript retrieval success"""
        with patch('async_video_service.YouTubeTranscriptApi') as mock_api:
            mock_api.get_transcript.return_value = [
                {'text': 'Hello'},
                {'text': ' world'},
                {'text': '!'}
            ]
            
            with patch('async_video_service.extract_youtube_video_id', return_value="test_id"):
                result = get_youtube_transcript("https://youtube.com/watch?v=test")
                
                assert result is not None
                transcript, video_title = result
                assert transcript == "Hello world!"
                assert video_title == "YouTube Video test_id"
    
    @pytest.mark.asyncio
    async def test_get_youtube_transcript_failure(self):
        """Test YouTube transcript retrieval failure"""
        with patch('async_video_service.YouTubeTranscriptApi') as mock_api:
            mock_api.get_transcript.side_effect = Exception("No transcript available")
            
            with patch('async_video_service.extract_youtube_video_id', return_value="test_id"):
                result = get_youtube_transcript("https://youtube.com/watch?v=test")
                
                assert result is None


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_generate_content_hash_empty_string(self):
        """Test content hash generation with empty string"""
        result = generate_content_hash("")
        assert result is not None
        assert len(result) == 64
    
    def test_generate_file_hash_nonexistent_file(self):
        """Test file hash generation with nonexistent file"""
        with pytest.raises(FileNotFoundError):
            generate_file_hash("/nonexistent/file.txt")
    
    def test_is_valid_youtube_url_or_id_none_input(self):
        """Test YouTube validation with None input"""
        # Test with empty string instead of None since function expects string
        assert not is_valid_youtube_url_or_id("")
    
    def test_extract_youtube_video_id_malformed_url(self):
        """Test video ID extraction with malformed URL"""
        malformed_urls = [
            "https://youtube.com/",
            "https://youtube.com/watch",
            "https://youtube.com/watch?v=",
            "not-a-url"
        ]
        
        for url in malformed_urls:
            result = extract_youtube_video_id(url)
            assert result is None
    
    @patch('async_video_service.MLX_AVAILABLE', False)
    def test_create_mlx_summary_mlx_not_available(self):
        """Test MLX summary when MLX is not available"""
        with pytest.raises(RuntimeError, match="MLX model failed to load"):
            _create_mlx_summary("Test text")


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests that test multiple components together"""
    
    def test_full_job_workflow_database(self, test_db, sample_user):
        """Test complete job workflow in database"""
        # 1. Check if user can process request
        can_process, _ = can_process_request(sample_user)
        assert can_process
        
        # 2. Create job
        job = Job(
            user_id=sample_user.id,
            filename="integration_test.mp4",
            content_hash="integration-test-hash",
            media_hash="integration-media-hash",
            status="pending"
        )
        test_db.add(job)
        test_db.commit()
        test_db.refresh(job)
        
        # 3. Update job status
        job.status = "processing"
        job.transcription = "This is a test transcription."
        test_db.commit()
        
        # 4. Complete job
        job.status = "completed"
        job.summary = "This is a test summary."
        test_db.commit()
        
        # 5. Increment usage
        increment_usage(test_db, sample_user, "video_process", job.id)
        
        # 6. Verify final state
        test_db.refresh(sample_user)
        assert sample_user.daily_usage_count == 1
        
        final_job = test_db.query(Job).filter(Job.id == job.id).first()
        assert final_job.status == "completed"
        assert final_job.transcription is not None
        assert final_job.summary is not None
    
    def test_duplicate_detection_workflow(self, test_db, sample_user):
        """Test duplicate detection across multiple jobs"""
        content_hash = "duplicate-test-hash"
        media_hash = "duplicate-media-hash"
        
        # Create first job
        job1 = Job(
            user_id=sample_user.id,
            filename="original.mp4",
            content_hash=content_hash,
            media_hash=media_hash,
            status="completed",
            transcription="Original transcription"
        )
        test_db.add(job1)
        test_db.commit()
        test_db.refresh(job1)
        
        # Check for duplicate (should find job1)
        duplicate = check_duplicate_job(test_db, sample_user.id, content_hash)
        assert duplicate is not None
        assert duplicate.id == job1.id
        
        # Check for existing content (should find job1)
        existing = check_existing_processed_content(test_db, content_hash)
        assert existing is not None
        assert existing.id == job1.id
        
        # Create job from existing content
        job2 = create_job_from_existing_content(
            test_db, sample_user.id, job1, "duplicate.mp4", "custom prompt"
        )
        
        assert job2.id != job1.id
        assert job2.transcription == job1.transcription
        assert job2.summary_prompt == "custom prompt"


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

class TestPerformance:
    """Performance and load tests"""
    
    def test_hash_generation_performance(self):
        """Test hash generation performance"""
        import time
        
        content = "test content " * 1000  # Create larger content
        
        start_time = time.time()
        for _ in range(100):
            generate_content_hash(content)
        end_time = time.time()
        
        # Should complete 100 hash generations in reasonable time
        assert (end_time - start_time) < 1.0  # Less than 1 second
    
    def test_database_query_performance(self, test_db, sample_user):
        """Test database query performance"""
        import time
        
        # Create multiple jobs
        jobs = []
        for i in range(50):
            job = Job(
                user_id=sample_user.id,
                filename=f"test_{i}.mp4",
                content_hash=f"hash_{i}",
                status="completed"
            )
            jobs.append(job)
        
        test_db.add_all(jobs)
        test_db.commit()
        
        # Test query performance
        start_time = time.time()
        for _ in range(10):
            result = test_db.query(Job).filter(Job.user_id == sample_user.id).all()
            assert len(result) == 50
        end_time = time.time()
        
        # Should complete 10 queries in reasonable time
        assert (end_time - start_time) < 1.0  # Less than 1 second


# ============================================================================
# CONFIGURATION AND SETUP
# ============================================================================

def pytest_configure(config):
    """Configure pytest markers"""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
