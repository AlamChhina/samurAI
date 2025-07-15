from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Request, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import asyncio
import platform
import moviepy as mp
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import edge_tts
import os
import torch
import tempfile
import shutil
from pathlib import Path
import uuid
import hashlib
from typing import Optional
from sqlalchemy.orm import Session
from database import get_db, User, Job
import aiofiles
from datetime import datetime, timezone
import httpx
import jwt
from pydantic import BaseModel
from dotenv import load_dotenv
import yt_dlp
import re
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
from transformers import pipeline

# Python 3.13 compatibility for audio modules
try:
    import aifc
except ImportError:
    try:
        import standard_aifc as aifc
    except ImportError:
        aifc = None

try:
    import chunk
except ImportError:
    try:
        import standard_chunk as chunk
    except ImportError:
        chunk = None

try:
    import sunau
except ImportError:
    try:
        import standard_sunau as sunau
    except ImportError:
        sunau = None

# Load environment variables
load_dotenv()

# Constants
AUTH_REQUIRED_MSG = "Authentication required"
FILE_NOT_FOUND_MSG = "File not found"

# Initialize FastAPI app
app = FastAPI(title="Video Audio Text Service", version="2.0.0")

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Configuration
MODEL_ID = "openai/whisper-base"
SUMMARIZATION_MODEL_ID = "sshleifer/distilbart-cnn-12-6"  # Lightweight BART model for summarization
OUTPUT_DIR = Path("./outputs")
TRANSCRIPTS_DIR = OUTPUT_DIR / "transcripts"
SPEECH_DIR = OUTPUT_DIR / "speech"
SUMMARIES_DIR = OUTPUT_DIR / "summaries"

# Google OAuth settings
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "your-google-client-id")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "your-google-client-secret")
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:8000/auth/callback")

# Edge-TTS settings (high-quality neural voices)
PREFERRED_VOICE = os.getenv("PREFERRED_VOICE", "en-US-AriaNeural")  # High-quality edge-tts voice

print("🔊 TTS Engine: Edge-TTS (Microsoft Neural Voices) ✅")
print("📝 Summarization: DistilBART (Lightweight & Efficient) ✅")

# Debug: Print loaded values (remove in production)
print("🔍 Debug - Loaded OAuth config:")
print(f"   CLIENT_ID: {GOOGLE_CLIENT_ID[:20]}..." if len(GOOGLE_CLIENT_ID) > 20 else f"   CLIENT_ID: {GOOGLE_CLIENT_ID}")
print(f"   CLIENT_SECRET: {GOOGLE_CLIENT_SECRET[:10]}..." if len(GOOGLE_CLIENT_SECRET) > 10 else f"   CLIENT_SECRET: {GOOGLE_CLIENT_SECRET}")
print(f"   REDIRECT_URI: {GOOGLE_REDIRECT_URI}")

# Validate OAuth configuration
if GOOGLE_CLIENT_ID == "your-google-client-id" or GOOGLE_CLIENT_SECRET == "your-google-client-secret":
    print("⚠️  WARNING: Google OAuth not configured properly!")
    print("   Please update your .env file with:")
    print("   - GOOGLE_CLIENT_ID (from Google Cloud Console)")
    print("   - GOOGLE_CLIENT_SECRET (from Google Cloud Console)")
    print("   - Ensure redirect URI is added to Google Cloud Console: http://localhost:8000/auth/callback")
else:
    print("✅ Google OAuth configuration loaded successfully")

# Create output directories
TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
SPEECH_DIR.mkdir(parents=True, exist_ok=True)
SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)

# Set up device for Apple Silicon optimization
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple Silicon GPU (MPS)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA GPU")
else:
    device = torch.device("cpu")
    print("Using CPU")

# Global variables for model caching
processor = None
model = None
summarizer = None
security = HTTPBearer(auto_error=False)

class JobResponse(BaseModel):
    id: str
    filename: str
    status: str
    created_at: datetime
    transcription: Optional[str] = None
    transcript_file: Optional[str] = None
    speech_file: Optional[str] = None
    summary: Optional[str] = None
    summary_file: Optional[str] = None
    summary_speech_file: Optional[str] = None
    summary_prompt: Optional[str] = None

class YouTubeRequest(BaseModel):
    url: str
    summary_prompt: Optional[str] = None

class VideoUploadRequest(BaseModel):
    summary_prompt: Optional[str] = None

class SummaryRequest(BaseModel):
    job_id: str
    summary_prompt: str

def load_model():
    """Load Whisper model and processor (cached)"""
    global processor, model
    if processor is None or model is None:
        print("Loading Whisper model...")
        processor = WhisperProcessor.from_pretrained(MODEL_ID)
        model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)
        
        # Configure model to handle attention masks properly
        model.config.use_cache = True
        model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")
        
        model = model.to(device)
        print(f"Model loaded on device: {next(model.parameters()).device}")
    return processor, model

async def get_current_user(request: Request, db: Session = Depends(get_db)):
    """Get current user from cookie or authorization header"""
    token = None
    
    # Try to get token from cookie first
    token = request.cookies.get("access_token")
    
    # If not in cookie, try authorization header
    if not token:
        auth_header = request.headers.get("authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
    
    if not token:
        return None
    
    try:
        # For simplicity, we're using email as token
        # In production, use proper JWT verification
        user = await asyncio.to_thread(lambda: db.query(User).filter(User.email == token).first())
        return user
    except Exception:
        return None

def is_valid_youtube_url_or_id(input_str: str) -> bool:
    """Validate if the input is a valid YouTube URL or video ID"""
    # Check if it's a valid YouTube URL
    youtube_url_patterns = [
        r'(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/',
        r'(https?://)?youtu\.be/',
        r'(https?://)?(www\.)?youtube\.com/watch\?v=',
        r'(https?://)?(www\.)?youtube\.com/embed/',
        r'(https?://)?(www\.)?youtube\.com/v/',
    ]
    
    for pattern in youtube_url_patterns:
        if re.search(pattern, input_str, re.IGNORECASE):
            return True
    
    # Check if it's a valid video ID (11 characters, alphanumeric with - and _)
    video_id_pattern = r'^[a-zA-Z0-9_-]{11}$'
    if re.match(video_id_pattern, input_str.strip()):
        return True
    
    return False

def is_valid_youtube_url(url: str) -> bool:
    """Validate if the URL is a valid YouTube URL (backward compatibility)"""
    return is_valid_youtube_url_or_id(url)

def generate_content_hash(content: str) -> str:
    """Generate a hash for content to detect duplicates"""
    return hashlib.sha256(content.encode('utf-8')).hexdigest()

def generate_file_hash(file_path: str) -> str:
    """Generate a hash for a file to detect duplicates"""
    hash_obj = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_obj.update(chunk)
    return hash_obj.hexdigest()

async def generate_file_hash_async(file_path: str) -> str:
    """Generate a hash for a file asynchronously"""
    def _hash_file():
        return generate_file_hash(file_path)
    return await asyncio.to_thread(_hash_file)

def check_duplicate_job(db: Session, user_id: str, content_hash: str) -> Optional[Job]:
    """Check if a job with the same content hash already exists for the user"""
    return db.query(Job).filter(
        Job.user_id == user_id,
        Job.content_hash == content_hash,
        Job.status.in_(["completed", "processing", "pending"])
    ).first()

async def download_youtube_video(url: str) -> Optional[tuple[str, str]]:
    """Download YouTube video and return the local file path"""
    try:
        def _download():
            # Create temporary filename
            temp_filename = f"temp_youtube_{uuid.uuid4().hex}"
            
            # yt-dlp options for best quality video with audio
            ydl_opts = {
                'format': 'best[ext=mp4]/best',  # Prefer mp4, fallback to best available
                'outtmpl': f'{temp_filename}.%(ext)s',
                'quiet': False,  # Set to True to suppress output
                'no_warnings': False,
                'extractaudio': False,  # We want video with audio
                'audioformat': 'mp3',
                'embed_subs': False,
                'writesubtitles': False,
                'writeautomaticsub': False,
            }
            
            print(f"Downloading YouTube video: {url}")
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Extract info to get the title and filename
                info = ydl.extract_info(url, download=False)
                title = info.get('title', 'Unknown')
                print(f"Video title: {title}")
                
                # Download the video
                ydl.download([url])
                
                # Find the downloaded file (yt-dlp may change the extension)
                import glob
                downloaded_files = glob.glob(f"{temp_filename}.*")
                
                if downloaded_files:
                    downloaded_file = downloaded_files[0]
                    print(f"Downloaded: {downloaded_file}")
                    return downloaded_file, title
                else:
                    print("No file was downloaded")
                    return None, None
        
        result = await asyncio.to_thread(_download)
        return result
    except Exception as e:
        print(f"Error downloading YouTube video: {str(e)}")
        return None, None

async def extract_audio_from_video(video_path: str) -> Optional[str]:
    """Extract audio from video file and save as temporary WAV file"""
    try:
        def _extract():
            print(f"Attempting to extract audio from: {video_path}")
            video = mp.VideoFileClip(video_path)
            print(f"Video loaded successfully. Duration: {video.duration}s")
            
            audio = video.audio
            if audio is None:
                print("No audio track found in video")
                video.close()
                return None
                
            temp_audio_path = f"temp_audio_{uuid.uuid4().hex}.wav"
            print(f"Extracting audio to: {temp_audio_path}")
            
            audio.write_audiofile(temp_audio_path)
            audio.close()
            video.close()
            return temp_audio_path
        
        result = await asyncio.to_thread(_extract)
        if result:
            print("Audio extraction completed.")
        return result
    except Exception as e:
        print(f"Error extracting audio: {str(e)}")
        return None

async def convert_audio_to_wav(audio_path: str) -> Optional[str]:
    """Convert audio file to WAV format using moviepy"""
    try:
        def _convert():
            print(f"Converting audio file: {audio_path}")
            
            # Use moviepy to convert audio
            from moviepy.editor import AudioFileClip
            
            audio_clip = AudioFileClip(audio_path)
            temp_wav_path = f"temp_audio_{uuid.uuid4().hex}.wav"
            
            print(f"Converting to WAV: {temp_wav_path}")
            audio_clip.write_audiofile(temp_wav_path, verbose=False, logger=None)
            audio_clip.close()
            
            print(f"Audio conversion completed. Output file size: {os.path.getsize(temp_wav_path)} bytes")
            return temp_wav_path
        
        result = await asyncio.to_thread(_convert)
        return result
        
    except Exception as e:
        print(f"Error converting audio: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

async def transcribe_audio(audio_path: str) -> Optional[str]:
    """Transcribe audio using Whisper - handles full length audio by chunking"""
    try:
        def _transcribe():
            import librosa
            import warnings
            
            # Suppress the specific attention mask warning
            warnings.filterwarnings("ignore", message=".*attention_mask.*")
            
            # Load model
            processor, model = load_model()
            
            # Load audio file
            audio, _ = librosa.load(audio_path, sr=16000)
            audio_duration = len(audio) / 16000
            print(f"Audio duration: {audio_duration:.2f} seconds")
            
            # Process audio in chunks
            chunk_length = 30  # seconds
            chunk_samples = chunk_length * 16000
            transcriptions = []
            
            for i in range(0, len(audio), chunk_samples):
                chunk = audio[i:i + chunk_samples]
                chunk_num = i // chunk_samples + 1
                total_chunks = (len(audio) + chunk_samples - 1) // chunk_samples
                
                print(f"Processing chunk {chunk_num}/{total_chunks}")
                
                # Process chunk with proper attention mask
                inputs = processor(chunk, sampling_rate=16000, return_tensors="pt")
                inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                
                # Create attention mask if not present
                if "attention_mask" not in inputs and "input_features" in inputs:
                    # For Whisper, create attention mask based on input_features
                    batch_size, _, seq_len = inputs["input_features"].shape
                    inputs["attention_mask"] = torch.ones((batch_size, seq_len), device=device, dtype=torch.long)
                
                # Generate transcription with forced parameters
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs, 
                        max_new_tokens=224,
                        language="en",
                        task="transcribe",
                        do_sample=False,  # Use deterministic generation
                        num_beams=1,      # Use greedy decoding
                        pad_token_id=processor.tokenizer.eos_token_id,
                        eos_token_id=processor.tokenizer.eos_token_id,
                        use_cache=True
                    )
                
                chunk_transcription = processor.batch_decode(outputs, skip_special_tokens=True)[0]
                transcriptions.append(chunk_transcription)
            
            full_transcription = " ".join(transcriptions)
            print(f"Transcribed {len(transcriptions)} chunks")
            return full_transcription
        
        result = await asyncio.to_thread(_transcribe)
        return result
    except Exception as e:
        print(f"Error during transcription: {e}")
        return None

async def generate_edge_tts(text: str, output_path: str, voice: str = None) -> bool:
    """Generate speech using edge-tts (Microsoft TTS) with high-quality neural voices"""
    try:
        # High-quality neural voices - these sound very natural
        premium_voices = [
            "en-US-JennyMultilingualNeural",  # Very natural, multilingual
            "en-US-AriaNeural",               # Professional, clear
            "en-US-DavisNeural",              # Natural male voice
            "en-US-AmberNeural",              # Warm female voice
            "en-US-BrandonNeural",            # Deep male voice
            "en-US-ChristopherNeural",        # Calm male voice
            "en-US-CoraNeural",               # Energetic female voice
            "en-US-ElizabethNeural",          # Sophisticated female voice
            "en-US-JaneNeural",               # Professional female voice
            "en-US-JasonNeural",              # Confident male voice
            "en-US-MichelleNeural",           # Pleasant female voice
            "en-US-MonicaNeural",             # Clear female voice
            "en-US-SaraNeural",               # Friendly female voice
            "en-US-TonyNeural"                # Professional male voice
        ]
        
        # Use specified voice or default to highest quality
        selected_voice = voice if voice else premium_voices[0]
        
        # Ensure text is not empty and is properly formatted
        if not text or not text.strip():
            print("❌ Empty text provided for TTS")
            return False
        
        # Clean the text
        clean_text = text.strip()
        if len(clean_text) > 32767:  # Edge-TTS limit
            clean_text = clean_text[:32767]
            print(f"⚠️ Text truncated to {len(clean_text)} characters for edge-tts")
        
        print(f"🎤 Using voice: {selected_voice}")
        communicate = edge_tts.Communicate(clean_text, selected_voice)
        await communicate.save(output_path)
        
        # Verify file was created and has content
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            print(f"✅ High-quality speech generated using edge-tts: {output_path}")
            return True
        else:
            print("❌ Edge-TTS generated empty file")
            return False
            
    except Exception as e:
        print(f"❌ Edge-TTS failed: {e}")
        # Try with a different voice as fallback
        if voice != "en-US-AriaNeural":
            print("🔄 Retrying with AriaNeural voice...")
            return await generate_edge_tts(text, output_path, "en-US-AriaNeural")
        return False

async def generate_system_tts(text: str, output_path: str) -> bool:
    """Generate speech using system TTS (macOS 'say' command)"""
    try:
        if platform.system() != "Darwin":
            return False
            
        def _system_tts():
            import subprocess
            aiff_path = output_path.replace('.mp3', '.aiff')
            subprocess.run(["say", "-o", aiff_path, text], check=True)
            
            # Convert AIFF to MP3 if ffmpeg is available
            if output_path.endswith('.mp3') and aiff_path != output_path:
                try:
                    subprocess.run([
                        "ffmpeg", "-i", aiff_path, "-acodec", "mp3", output_path, "-y"
                    ], check=True, capture_output=True)
                    os.remove(aiff_path)
                except (subprocess.CalledProcessError, FileNotFoundError):
                    # Keep AIFF file if ffmpeg not available
                    if aiff_path != output_path:
                        final_path = output_path.replace('.mp3', '.aiff')
                        os.rename(aiff_path, final_path)
            
            print("Speech generated using system TTS")
            return True
        
        result = await asyncio.to_thread(_system_tts)
        return result
    except Exception as e:
        print(f"System TTS failed: {e}")
        return False

async def text_to_speech(text: str, output_path: str) -> bool:
    """Convert text to speech using edge-tts (Microsoft Neural Voices)"""
    print("🎵 Starting high-quality text-to-speech generation with edge-tts...")
    
    # Try edge-tts with premium neural voices
    print("🎤 Using edge-tts with premium neural voices...")
    if await generate_edge_tts(text, output_path, PREFERRED_VOICE):
        return True
    
    # Final fallback to system TTS on macOS
    print("⚠️ Attempting system TTS fallback...")
    if await generate_system_tts(text, output_path):
        return True
    
    print("❌ All TTS engines failed")
    return False

async def process_video_background(job_id: str, video_path: str, filename: str, db: Session, custom_prompt: Optional[str] = None):
    """Background task to process video"""
    job = db.query(Job).filter(Job.id == job_id).first()
    
    try:
        # Update status to processing
        job.status = "processing"
        db.commit()
        
        # Extract audio
        audio_path = await extract_audio_from_video(video_path)
        if not audio_path:
            raise RuntimeError("Failed to extract audio")
        
        # Transcribe audio
        transcription = await transcribe_audio(audio_path)
        if not transcription:
            raise RuntimeError("Failed to transcribe audio")
        
        # Save transcription
        video_basename = Path(filename).stem
        transcript_filename = f"{video_basename}_{job_id}_transcript.txt"
        transcript_path = TRANSCRIPTS_DIR / transcript_filename
        
        async with aiofiles.open(transcript_path, 'w') as f:
            await f.write(transcription)
        
        # Generate speech
        speech_filename = f"{video_basename}_{job_id}_speech.mp3"
        speech_path = SPEECH_DIR / speech_filename
        success = await text_to_speech(transcription, str(speech_path))
        
        if not success:
            raise RuntimeError("Failed to generate speech")
        
        # Generate summary using DistilBART
        print("📝 Generating summary using DistilBART...")
        summary = await generate_summary(transcription, custom_prompt)
        
        summary_filename = None
        summary_speech_filename = None
        
        if summary:
            # Save summary
            summary_filename = f"{video_basename}_{job_id}_summary.txt"
            summary_path = SUMMARIES_DIR / summary_filename
            
            async with aiofiles.open(summary_path, 'w') as f:
                await f.write(summary)
            
            # Generate speech for summary
            print("🎵 Generating speech for summary...")
            summary_speech_filename = f"{video_basename}_{job_id}_summary_speech.mp3"
            summary_speech_path = SPEECH_DIR / summary_speech_filename
            summary_speech_success = await text_to_speech(summary, str(summary_speech_path))
            
            if not summary_speech_success:
                print("⚠️ Failed to generate summary speech, but continuing...")
                summary_speech_filename = None
        
        # Update job with results
        job.status = "completed"
        job.transcription = transcription
        job.transcript_file = transcript_filename
        job.speech_file = speech_filename
        job.summary = summary
        job.summary_file = summary_filename
        job.summary_speech_file = summary_speech_filename
        job.summary_prompt = custom_prompt
        job.completed_at = datetime.now(timezone.utc)
        db.commit()
        
        # Clean up
        if os.path.exists(audio_path):
            os.remove(audio_path)
        
    except Exception as e:
        job.status = "failed"
        job.error_message = str(e)
        db.commit()
    
    finally:
        # Clean up video file
        if os.path.exists(video_path):
            os.remove(video_path)

async def process_audio_background(job_id: str, audio_path: str, filename: str, db: Session, summary_prompt: Optional[str] = None):
    """Background task to process audio"""
    job = db.query(Job).filter(Job.id == job_id).first()
    
    try:
        # Update status to processing
        job.status = "processing"
        db.commit()
        
        # Convert audio to WAV format if needed
        file_extension = Path(filename).suffix.lower()
        processed_audio_path = audio_path
        
        if file_extension != '.wav':
            print(f"Converting {filename} to WAV format...")
            processed_audio_path = await convert_audio_to_wav(audio_path)
            if not processed_audio_path:
                raise RuntimeError("Failed to convert audio to WAV format")
        
        # Transcribe audio
        transcription = await transcribe_audio(processed_audio_path)
        if not transcription:
            raise RuntimeError("Failed to transcribe audio")
        
        # Save transcription
        audio_basename = Path(filename).stem
        transcript_filename = f"{audio_basename}_{job_id}_transcript.txt"
        transcript_path = TRANSCRIPTS_DIR / transcript_filename
        
        async with aiofiles.open(transcript_path, 'w') as f:
            await f.write(transcription)
        
        # Generate speech
        speech_filename = f"{audio_basename}_{job_id}_speech.mp3"
        speech_path = SPEECH_DIR / speech_filename
        success = await text_to_speech(transcription, str(speech_path))
        
        if not success:
            raise RuntimeError("Failed to generate speech")
        
        # Generate summary using DistilBART
        print("📝 Generating summary using DistilBART...")
        summary = await generate_summary(transcription, summary_prompt)
        
        summary_filename = None
        summary_speech_filename = None
        
        if summary:
            # Save summary
            summary_filename = f"{audio_basename}_{job_id}_summary.txt"
            summary_path = SUMMARIES_DIR / summary_filename
            
            async with aiofiles.open(summary_path, 'w') as f:
                await f.write(summary)
            
            # Generate summary speech
            summary_speech_filename = f"{audio_basename}_{job_id}_summary_speech.mp3"
            summary_speech_path = SPEECH_DIR / summary_speech_filename
            summary_speech_success = await text_to_speech(summary, str(summary_speech_path))
            
            if not summary_speech_success:
                print("⚠️ Failed to generate summary speech, but continuing...")
                summary_speech_filename = None
        
        # Update job with results
        job.status = "completed"
        job.transcription = transcription
        job.transcript_file = transcript_filename
        job.speech_file = speech_filename
        job.summary = summary
        job.summary_file = summary_filename
        job.summary_speech_file = summary_speech_filename
        job.summary_prompt = summary_prompt
        job.completed_at = datetime.now(timezone.utc)
        db.commit()
        
        # Clean up processed audio file if it's different from original
        if processed_audio_path != audio_path and os.path.exists(processed_audio_path):
            os.remove(processed_audio_path)
        
    except Exception as e:
        job.status = "failed"
        job.error_message = str(e)
        db.commit()
    
    finally:
        # Clean up original audio file
        if os.path.exists(audio_path):
            os.remove(audio_path)

async def process_youtube_background(job_id: str, youtube_url: str, db: Session, custom_prompt: Optional[str] = None):
    """Background task to process YouTube video"""
    job = db.query(Job).filter(Job.id == job_id).first()
    video_path = None
    audio_path = None
    
    try:
        # Update status to processing
        job.status = "processing"
        db.commit()
        
        # Fast path: Try to get transcript directly from YouTube API
        print("🚀 Attempting fast transcript path using YouTube Transcript API...")
        transcript_result = get_youtube_transcript(youtube_url)
        
        if transcript_result:
            # Fast path successful!
            transcription, video_title = transcript_result
            print(f"✅ Fast transcript successful for '{video_title}'")
            
            # Update job filename with actual video title
            job.filename = f"YouTube: {video_title}"
            db.commit()
            
        else:
            # Fall back to normal flow: download + Whisper
            print("⬇️ Fast transcript not available, falling back to download + Whisper...")
            
            # Download YouTube video
            download_result = await download_youtube_video(youtube_url)
            if not download_result:
                raise RuntimeError("Failed to download YouTube video")
            
            video_path, video_title = download_result
            
            # Update job filename with actual video title
            job.filename = f"YouTube: {video_title}"
            db.commit()
            
            # Extract audio
            audio_path = await extract_audio_from_video(video_path)
            if not audio_path:
                raise RuntimeError("Failed to extract audio")
            
            # Transcribe audio
            transcription = await transcribe_audio(audio_path)
            if not transcription:
                raise RuntimeError("Failed to transcribe audio")
                
            print(f"✅ Normal transcript flow completed for '{video_title}'")
        
        # Common processing for both paths: Save transcription and generate speech
        safe_title = re.sub(r'[^\w\s-]', '', video_title).strip()[:50]  # Safe filename
        transcript_filename = f"{safe_title}_{job_id}_transcript.txt"
        transcript_path = TRANSCRIPTS_DIR / transcript_filename
        
        async with aiofiles.open(transcript_path, 'w') as f:
            await f.write(transcription)
        
        # Generate summary using DistilBART
        print("📝 Generating summary using DistilBART...")
        summary = await generate_summary(transcription, custom_prompt)
        
        summary_filename = None
        summary_speech_filename = None
        
        if summary:
            # Save summary
            summary_filename = f"{safe_title}_{job_id}_summary.txt"
            summary_path = SUMMARIES_DIR / summary_filename
            
            async with aiofiles.open(summary_path, 'w') as f:
                await f.write(summary)
            
            # Generate speech for summary
            print("🎵 Generating speech for summary...")
            summary_speech_filename = f"{safe_title}_{job_id}_summary_speech.mp3"
            summary_speech_path = SPEECH_DIR / summary_speech_filename
            summary_speech_success = await text_to_speech(summary, str(summary_speech_path))
            
            if not summary_speech_success:
                print("⚠️ Failed to generate summary speech, but continuing...")
                summary_speech_filename = None
        
        # Generate speech for full transcript
        print("🎵 Generating speech for full transcript...")
        speech_filename = f"{safe_title}_{job_id}_speech.mp3"
        speech_path = SPEECH_DIR / speech_filename
        success = await text_to_speech(transcription, str(speech_path))
        
        if not success:
            raise RuntimeError("Failed to generate speech")
        
        # Update job with results
        job.status = "completed"
        job.transcription = transcription
        job.transcript_file = transcript_filename
        job.speech_file = speech_filename
        job.summary = summary
        job.summary_file = summary_filename
        job.summary_speech_file = summary_speech_filename
        job.summary_prompt = custom_prompt
        job.completed_at = datetime.now(timezone.utc)
        db.commit()
        
        # Clean up audio file if it exists
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)
        
        print(f"🎉 Job {job_id} completed successfully!")
        
    except Exception as e:
        print(f"❌ Job {job_id} failed: {e}")
        job.status = "failed"
        job.error_message = str(e)
        db.commit()
    
    finally:
        # Clean up temporary files
        if video_path and os.path.exists(video_path):
            os.remove(video_path)
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)

def extract_youtube_video_id(url_or_id: str) -> Optional[str]:
    """Extract YouTube video ID from URL or return the ID if it's already a video ID"""
    # First check if it's already a valid video ID (11 characters, alphanumeric with - and _)
    video_id_pattern = r'^[a-zA-Z0-9_-]{11}$'
    if re.match(video_id_pattern, url_or_id.strip()):
        return url_or_id.strip()
    
    # If not a plain ID, try to extract from URL
    patterns = [
        r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/|youtube\.com/v/)([^&\n?#]+)',
        r'youtube\.com/watch\?.*v=([^&\n?#]+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url_or_id, re.IGNORECASE)
        if match:
            return match.group(1)
    return None

def normalize_youtube_input(url_or_id: str) -> str:
    """Convert video ID to full YouTube URL or return the URL as-is"""
    # Check if it's a plain video ID
    video_id_pattern = r'^[a-zA-Z0-9_-]{11}$'
    if re.match(video_id_pattern, url_or_id.strip()):
        return f"https://www.youtube.com/watch?v={url_or_id.strip()}"
    
    # If it's already a URL, return as-is
    return url_or_id

def get_youtube_transcript(url: str) -> Optional[tuple[str, str]]:
    """
    Try to get YouTube transcript directly using YouTube Transcript API.
    Returns (transcript_text, video_title) if successful, None if not available.
    """
    try:
        # Extract video ID
        video_id = extract_youtube_video_id(url)
        if not video_id:
            print(f"Could not extract video ID from URL: {url}")
            return None
        
        print(f"Attempting to get transcript for video ID: {video_id}")
        
        # Try to get transcript
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'en-US', 'en-GB'])
        
        # Combine transcript entries into single text
        transcript_text = ' '.join([entry['text'] for entry in transcript_list])
        
        if not transcript_text.strip():
            print("Transcript is empty")
            return None
        
        # Get video title using yt-dlp (lightweight info extraction)
        try:
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'extract_flat': True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                video_title = info.get('title', f'YouTube Video {video_id}')
        except Exception as e:
            print(f"Error getting video title: {e}")
            video_title = f'YouTube Video {video_id}'
        
        print(f"Successfully obtained transcript for '{video_title}' ({len(transcript_text)} characters)")
        return transcript_text, video_title
        
    except (TranscriptsDisabled, NoTranscriptFound) as e:
        print(f"No transcript available for {url}: {e}")
        return None
    except Exception as e:
        print(f"Error getting transcript for {url}: {e}")
        return None

# Custom exceptions
class VideoProcessingError(Exception):
    """Custom exception for video processing errors"""
    pass

class AudioExtractionError(VideoProcessingError):
    """Exception for audio extraction failures"""
    pass

class TranscriptionError(VideoProcessingError):
    """Exception for transcription failures"""
    pass

class SpeechGenerationError(VideoProcessingError):
    """Exception for speech generation failures"""
    pass

# Routes
@app.get("/", response_class=HTMLResponse)
async def home(request: Request, user: User = Depends(get_current_user)):
    """Home page"""
    return templates.TemplateResponse("index.html", {"request": request, "user": user})

@app.get("/auth/google")
async def google_auth():
    """Redirect to Google OAuth"""
    google_auth_url = f"https://accounts.google.com/o/oauth2/auth?client_id={GOOGLE_CLIENT_ID}&redirect_uri={GOOGLE_REDIRECT_URI}&scope=openid email profile&response_type=code"
    return RedirectResponse(google_auth_url)

@app.get("/auth/callback")
async def google_callback(code: str, db: Session = Depends(get_db)):
    """Handle Google OAuth callback"""
    try:
        # Exchange code for token
        async with httpx.AsyncClient() as client:
            token_response = await client.post(
                "https://oauth2.googleapis.com/token",
                data={
                    "client_id": GOOGLE_CLIENT_ID,
                    "client_secret": GOOGLE_CLIENT_SECRET,
                    "code": code,
                    "grant_type": "authorization_code",
                    "redirect_uri": GOOGLE_REDIRECT_URI,
                }
            )
            token_data = token_response.json()
            
            # Check for errors in token response
            if "error" in token_data:
                print(f"OAuth token error: {token_data}")
                if token_data.get("error") == "invalid_client":
                    raise HTTPException(status_code=500, detail="Invalid Google OAuth configuration. Please check your client ID and secret.")
                raise HTTPException(status_code=400, detail=f"OAuth error: {token_data.get('error_description', 'Unknown error')}")
            
            if "access_token" not in token_data:
                raise HTTPException(status_code=400, detail="No access token received from Google")
            
            # Get user info
            user_response = await client.get(
                "https://www.googleapis.com/oauth2/v2/userinfo",
                headers={"Authorization": f"Bearer {token_data['access_token']}"}
            )
            user_data = user_response.json()
        
        # Create or get user
        user = db.query(User).filter(User.google_id == user_data["id"]).first()
        if not user:
            user = User(
                google_id=user_data["id"],
                email=user_data["email"],
                name=user_data["name"],
                picture=user_data.get("picture")
            )
            db.add(user)
            db.commit()
            db.refresh(user)
        
        # In production, create proper JWT token
        response = RedirectResponse("/dashboard")
        response.set_cookie("access_token", user.email)  # Simplified for demo
        return response
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Authentication failed: {str(e)}")

@app.get("/auth/logout")
async def logout():
    """Logout user"""
    response = RedirectResponse("/")
    response.delete_cookie("access_token")
    return response

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """User dashboard"""
    if not user:
        return RedirectResponse("/auth/google")
    
    jobs = db.query(Job).filter(Job.user_id == user.id).order_by(Job.created_at.desc()).all()
    return templates.TemplateResponse("dashboard.html", {"request": request, "user": user, "jobs": jobs})

@app.post("/api/process-video/")
async def process_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    summary_prompt: Optional[str] = None,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Process uploaded video file"""
    if not user:
        raise HTTPException(status_code=401, detail=AUTH_REQUIRED_MSG)
    
    # Validate file type
    valid_extensions = ['.mp4', '.mov', '.avi', '.mkv', '.wmv', '.flv', '.webm', '.m4v']
    file_extension = Path(file.filename).suffix.lower()
    
    if file_extension not in valid_extensions:
        raise HTTPException(status_code=400, detail=f"Supported formats: {', '.join(valid_extensions)}")
    
    # Save uploaded file temporarily to generate hash
    temp_video_path = f"temp_video_hash_{uuid.uuid4().hex}.{file_extension}"
    async with aiofiles.open(temp_video_path, 'wb') as f:
        content = await file.read()
        await f.write(content)
    
    # Generate content hash for duplicate detection
    file_hash = await generate_file_hash_async(temp_video_path)
    content_identifier = f"video:{file_hash}:{summary_prompt or 'default'}"
    content_hash = generate_content_hash(content_identifier)
    
    # Check for existing job with same content
    existing_job = check_duplicate_job(db, user.id, content_hash)
    if existing_job:
        # Clean up temporary file
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        return {
            "job_id": existing_job.id, 
            "status": existing_job.status, 
            "message": f"This video has already been processed (Job ID: {existing_job.id})",
            "duplicate": True
        }
    
    # Create job
    job = Job(
        user_id=user.id,
        filename=file.filename,
        content_hash=content_hash,
        status="pending",
        summary_prompt=summary_prompt
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    
    # Rename temporary file to job-specific name
    video_path = f"temp_video_{job.id}.{file_extension}"
    os.rename(temp_video_path, video_path)
    
    # Start background processing
    background_tasks.add_task(process_video_background, job.id, video_path, file.filename, db, summary_prompt)
    
    return {"job_id": job.id, "status": "pending", "message": "Video processing started"}

@app.post("/api/process-audio/")
async def process_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    summary_prompt: Optional[str] = None,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Process uploaded audio file and return transcription and speech"""
    if not user:
        raise HTTPException(status_code=401, detail=AUTH_REQUIRED_MSG)
    
    # Validate file type - check both content type and file extension
    valid_extensions = ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac', '.wma']
    file_extension = Path(file.filename).suffix.lower()
    
    if not (file.content_type and file.content_type.startswith('audio/')) and file_extension not in valid_extensions:
        raise HTTPException(status_code=400, detail=f"File must be an audio file. Supported formats: {', '.join(valid_extensions)}")
    
    # Save uploaded file temporarily to generate hash
    temp_audio_path = f"temp_audio_hash_{uuid.uuid4().hex}.{file_extension[1:]}"
    async with aiofiles.open(temp_audio_path, 'wb') as f:
        content = await file.read()
        await f.write(content)
    
    # Generate content hash for duplicate detection
    file_hash = await generate_file_hash_async(temp_audio_path)
    content_identifier = f"audio:{file_hash}:{summary_prompt or 'default'}"
    content_hash = generate_content_hash(content_identifier)
    
    # Check for existing job with same content
    existing_job = check_duplicate_job(db, user.id, content_hash)
    if existing_job:
        # Clean up temporary file
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        return {
            "job_id": existing_job.id, 
            "status": existing_job.status, 
            "message": f"This audio file has already been processed (Job ID: {existing_job.id})",
            "duplicate": True
        }
    
    # Create job in database
    job = Job(
        user_id=user.id,
        filename=file.filename,
        content_hash=content_hash,
        status="pending",
        summary_prompt=summary_prompt,
        created_at=datetime.now(timezone.utc)
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    
    # Rename temporary file to job-specific name
    audio_path = f"temp_audio_{job.id}.{file_extension[1:]}"  # Remove the dot from extension
    os.rename(temp_audio_path, audio_path)
    
    # Start background processing
    background_tasks.add_task(process_audio_background, job.id, audio_path, file.filename, db, summary_prompt)
    
    return {"job_id": job.id, "status": "pending", "message": "Audio processing started"}

@app.post("/api/process-youtube/")
async def process_youtube(
    background_tasks: BackgroundTasks,
    request: YouTubeRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Process YouTube video from URL or video ID"""
    if not user:
        raise HTTPException(status_code=401, detail=AUTH_REQUIRED_MSG)
    
    # Validate YouTube URL or video ID
    if not is_valid_youtube_url_or_id(request.url):
        raise HTTPException(status_code=400, detail="Invalid YouTube URL or video ID")
    
    # Normalize input to full URL format
    normalized_url = normalize_youtube_input(request.url)
    
    # Generate content hash for duplicate detection
    content_identifier = f"youtube:{extract_youtube_video_id(normalized_url)}:{request.summary_prompt or 'default'}"
    content_hash = generate_content_hash(content_identifier)
    
    # Check for existing job with same content
    existing_job = check_duplicate_job(db, user.id, content_hash)
    if existing_job:
        return {
            "job_id": existing_job.id, 
            "status": existing_job.status, 
            "message": f"This YouTube video has already been processed (Job ID: {existing_job.id})",
            "duplicate": True
        }
    
    # Create job with URL as filename initially
    job = Job(
        user_id=user.id,
        filename=f"YouTube: {normalized_url}",
        content_hash=content_hash,
        status="pending",
        summary_prompt=request.summary_prompt
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    
    # Start background processing for YouTube
    background_tasks.add_task(process_youtube_background, job.id, normalized_url, db, request.summary_prompt)
    
    return {"job_id": job.id, "status": "pending", "message": "YouTube video processing started"}

@app.get("/api/jobs/", response_model=list[JobResponse])
async def get_user_jobs(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Get user's jobs"""
    if not user:
        raise HTTPException(status_code=401, detail=AUTH_REQUIRED_MSG)
    
    jobs = db.query(Job).filter(Job.user_id == user.id).order_by(Job.created_at.desc()).all()
    return jobs

@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Get specific job"""
    if not user:
        raise HTTPException(status_code=401, detail=AUTH_REQUIRED_MSG)
    
    job = db.query(Job).filter(Job.id == job_id, Job.user_id == user.id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return job

@app.get("/download/transcript/{filename}")
async def download_transcript(filename: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Download transcript file"""
    if not user:
        raise HTTPException(status_code=401, detail=AUTH_REQUIRED_MSG)
    
    file_path = TRANSCRIPTS_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=FILE_NOT_FOUND_MSG)
    
    return FileResponse(file_path, filename=filename)

@app.get("/download/speech/{filename}")
async def download_speech(filename: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Download speech file"""
    if not user:
        raise HTTPException(status_code=401, detail=AUTH_REQUIRED_MSG)
    
    file_path = SPEECH_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=FILE_NOT_FOUND_MSG)
    
    return FileResponse(file_path, filename=filename)

@app.get("/download/summary/{filename}")
async def download_summary(filename: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Download summary file"""
    if not user:
        raise HTTPException(status_code=401, detail=AUTH_REQUIRED_MSG)
    
    file_path = SUMMARIES_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=FILE_NOT_FOUND_MSG)
    
    return FileResponse(file_path, filename=filename)

@app.get("/download/summary-speech/{filename}")
async def download_summary_speech(filename: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Download summary speech file"""
    if not user:
        raise HTTPException(status_code=401, detail=AUTH_REQUIRED_MSG)
    
    file_path = SPEECH_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=FILE_NOT_FOUND_MSG)
    
    return FileResponse(file_path, filename=filename)

@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Delete job and associated files"""
    if not user:
        raise HTTPException(status_code=401, detail=AUTH_REQUIRED_MSG)
    
    job = db.query(Job).filter(Job.id == job_id, Job.user_id == user.id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Delete associated files
    if job.transcript_file:
        transcript_path = TRANSCRIPTS_DIR / job.transcript_file
        if transcript_path.exists():
            transcript_path.unlink()
    
    if job.speech_file:
        speech_path = SPEECH_DIR / job.speech_file
        if speech_path.exists():
            speech_path.unlink()
    
    if job.summary_file:
        summary_path = SUMMARIES_DIR / job.summary_file
        if summary_path.exists():
            summary_path.unlink()
    
    if job.summary_speech_file:
        summary_speech_path = SPEECH_DIR / job.summary_speech_file
        if summary_speech_path.exists():
            summary_speech_path.unlink()
    
    if job.summary_file:
        summary_path = SUMMARIES_DIR / job.summary_file
        if summary_path.exists():
            summary_path.unlink()
    
    if job.summary_speech_file:
        summary_speech_path = SPEECH_DIR / job.summary_speech_file
        if summary_speech_path.exists():
            summary_speech_path.unlink()
    
    # Delete job from database
    db.delete(job)
    db.commit()
    
    return {"message": "Job deleted successfully"}

def load_summarizer():
    """Load DistilBART model for text summarization (cached)"""
    global summarizer
    if summarizer is None:
        print("Loading DistilBART summarization model...")
        try:
            # Use DistilBART for text summarization - lightweight and efficient
            if device.type == "mps":
                model_device = 0 if torch.backends.mps.is_available() else -1
            elif device.type == "cuda":
                model_device = 0
            else:
                model_device = -1  # CPU
                
            # Load the summarization pipeline with DistilBART
            summarizer = pipeline(
                "summarization",
                model=SUMMARIZATION_MODEL_ID,
                device=model_device,
                torch_dtype=torch.float16 if device.type in ["mps", "cuda"] else torch.float32,
                max_length=150,  # Maximum summary length
                min_length=50,   # Minimum summary length
                do_sample=False  # Use deterministic generation for consistent results
            )
            print(f"✅ DistilBART summarization model loaded on {device}")
        except Exception as e:
            print(f"❌ Failed to load DistilBART model: {e}")
            print("📝 Falling back to simple text truncation for summaries")
            summarizer = "fallback"
    return summarizer

def _summarize_chunk(summarizer_model, chunk: str, max_length: int = 100, min_length: int = 30) -> str:
    """Summarize a single chunk of text"""
    try:
        result = summarizer_model(chunk, max_length=max_length, min_length=min_length, do_sample=False)
        return result[0]['summary_text']
    except Exception as e:
        print(f"Chunk summarization failed: {e}")
        # Extract key sentences from this chunk as fallback
        sentences = chunk.split('. ')
        summary = '. '.join(sentences[:2])
        return summary

def _create_fallback_summary(text: str, num_sentences: int = 3) -> str:
    """Create a simple extractive summary by taking first N sentences"""
    sentences = text.split('. ')
    summary = '. '.join(sentences[:num_sentences])
    if not summary.endswith('.'):
        summary += '.'
    return summary

def _split_text_into_chunks(text: str, max_chunk_length: int = 1024) -> list[str]:
    """Split text into chunks for processing"""
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_chunk_length):
        chunk = ' '.join(words[i:i + max_chunk_length])
        chunks.append(chunk)
    return chunks

async def generate_summary(text: str, custom_prompt: Optional[str] = None) -> Optional[str]:
    """Generate a summary of the given text using DistilBART or custom prompt"""
    try:
        def _summarize():
            summarizer_model = load_summarizer()
            
            if summarizer_model == "fallback":
                return f"Summary (fallback): {_create_fallback_summary(text, 5)}"
            
            # If custom prompt is provided, use prompt-based summarization
            if custom_prompt:
                return _generate_custom_prompt_summary(text, custom_prompt)
            
            # Default behavior: use DistilBART for standard summarization
            max_chunk_length = 1024  # DistilBART's optimal input length
            words = text.split()
            
            # Short text, summarize directly
            if len(words) <= max_chunk_length:
                return _summarize_chunk(summarizer_model, text, max_length=150, min_length=50)
            
            # Long text, chunk and summarize
            chunks = _split_text_into_chunks(text, max_chunk_length)
            summaries = []
            
            for i, chunk in enumerate(chunks):
                print(f"Summarizing chunk {i+1}/{len(chunks)}")
                chunk_summary = _summarize_chunk(summarizer_model, chunk)
                summaries.append(chunk_summary)
            
            # Combine chunk summaries
            combined_summary = ' '.join(summaries)
            
            # If combined summary is still long, summarize it again
            if len(combined_summary.split()) > 200:
                return _summarize_chunk(summarizer_model, combined_summary, max_length=150, min_length=50)
            
            return combined_summary
        
        summary = await asyncio.to_thread(_summarize)
        prompt_type = "custom prompt" if custom_prompt else "standard"
        print(f"✅ Summary generated using {prompt_type} ({len(summary)} characters)")
        return summary
        
    except Exception as e:
        print(f"❌ Summary generation failed: {e}")
        return f"Summary (auto): {_create_fallback_summary(text)}"

def _generate_custom_prompt_summary(text: str, custom_prompt: str) -> str:
    """Generate summary using a custom prompt with intelligent extraction"""
    try:
        # For now, use guided extractive summarization with the custom prompt
        # This avoids loading additional models and works efficiently
        print(f"Generating custom summary with prompt: {custom_prompt[:50]}...")
        
        # Use the guided extractive approach which is fast and effective
        return _create_guided_extractive_summary(text, custom_prompt)
            
    except Exception as e:
        print(f"Error in custom prompt summarization: {e}")
        return _create_fallback_summary(text, 3)

def _create_guided_extractive_summary(text: str, prompt: str) -> str:
    """Create an intelligent extractive summary guided by the prompt"""
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    
    # Extract keywords from the prompt
    prompt_words = {word.lower().strip('.,!?;:') for word in prompt.split() 
                   if len(word) > 3 and word.lower() not in {'this', 'that', 'with', 'from', 'they', 'have', 'will', 'been', 'said', 'what', 'when', 'where', 'about', 'please', 'focus', 'summary', 'summarize'}}
    
    # Score sentences based on prompt relevance
    scored_sentences = []
    for sentence in sentences:
        sentence_words = {word.lower().strip('.,!?;:') for word in sentence.split()}
        
        # Calculate relevance score
        keyword_matches = len(prompt_words.intersection(sentence_words))
        sentence_length_score = min(len(sentence.split()) / 20, 1.0)  # Prefer medium-length sentences
        
        # Boost score for sentences containing multiple prompt keywords
        relevance_score = keyword_matches * 2 + sentence_length_score
        
        # Extra boost for sentences with specific focus words from prompt
        focus_words = {'ethical', 'concern', 'issue', 'problem', 'benefit', 'advantage', 'impact', 'effect', 'result', 'consequence'}
        if any(word in sentence.lower() for word in focus_words if word in prompt.lower()):
            relevance_score += 1
        
        scored_sentences.append((relevance_score, sentence))
    
    # Sort by relevance score and take top sentences
    scored_sentences.sort(key=lambda x: x[0], reverse=True)
    
    # Select top 3-5 sentences, ensuring we have enough content
    selected_sentences = []
    total_words = 0
    max_words = 150
    
    for score, sentence in scored_sentences:
        if score > 0 and total_words < max_words:  # Only include relevant sentences
            selected_sentences.append(sentence)
            total_words += len(sentence.split())
            if len(selected_sentences) >= 5:  # Max 5 sentences
                break
    
    # If no relevant sentences found, fall back to first few sentences
    if not selected_sentences:
        selected_sentences = sentences[:3]
    
    summary = '. '.join(selected_sentences)
    if not summary.endswith('.'):
        summary += '.'
    
    return f"Summary (guided by: {prompt[:30]}...): {summary}"

@app.post("/api/regenerate-summary/")
async def regenerate_summary(
    background_tasks: BackgroundTasks,
    request: SummaryRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Regenerate summary with custom prompt for existing job"""
    if not user:
        raise HTTPException(status_code=401, detail=AUTH_REQUIRED_MSG)
    
    # Get the job
    job = db.query(Job).filter(Job.id == request.job_id, Job.user_id == user.id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.status != "completed":
        raise HTTPException(status_code=400, detail="Job must be completed to regenerate summary")
    
    if not job.transcription:
        raise HTTPException(status_code=400, detail="No transcription available for this job")
    
    # Generate new summary in background
    async def regenerate_summary_background():
        try:
            print(f"🔄 Regenerating summary for job {job.id} with custom prompt...")
            
            # Generate new summary
            new_summary = await generate_summary(job.transcription, request.summary_prompt)
            
            if new_summary:
                # Save new summary
                safe_filename = re.sub(r'[^\w\s-]', '', job.filename).strip()[:50]
                summary_filename = f"{safe_filename}_{job.id}_summary_custom.txt"
                summary_path = SUMMARIES_DIR / summary_filename
                
                async with aiofiles.open(summary_path, 'w') as f:
                    await f.write(new_summary)
                
                # Generate speech for new summary
                summary_speech_filename = f"{safe_filename}_{job.id}_summary_custom_speech.mp3"
                summary_speech_path = SPEECH_DIR / summary_speech_filename
                speech_success = await text_to_speech(new_summary, str(summary_speech_path))
                
                # Update job with new summary
                job.summary = new_summary
                job.summary_file = summary_filename
                job.summary_speech_file = summary_speech_filename if speech_success else None
                job.summary_prompt = request.summary_prompt
                db.commit()
                
                print(f"✅ Summary regenerated successfully for job {job.id}")
            else:
                print(f"❌ Failed to regenerate summary for job {job.id}")
                
        except Exception as e:
            print(f"❌ Error regenerating summary: {e}")
    
    # Start regeneration in background
    background_tasks.add_task(regenerate_summary_background)
    
    return {"message": "Summary regeneration started", "job_id": job.id}
