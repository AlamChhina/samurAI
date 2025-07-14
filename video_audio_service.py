from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
import asyncio
import platform
import moviepy as mp
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from gtts import gTTS
import os
import torch
import tempfile
import shutil
from pathlib import Path
import uuid
from typing import Optional

# Initialize FastAPI app
app = FastAPI(title="Video Audio Text Service", version="1.0.0")

# Configuration
MODEL_ID = "openai/whisper-base"
OUTPUT_DIR = Path("./outputs")
TRANSCRIPTS_DIR = OUTPUT_DIR / "transcripts"
SPEECH_DIR = OUTPUT_DIR / "speech"

# Create output directories
TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
SPEECH_DIR.mkdir(parents=True, exist_ok=True)

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

def load_model():
    """Load Whisper model and processor (cached)"""
    global processor, model
    if processor is None or model is None:
        print("Loading Whisper model...")
        processor = WhisperProcessor.from_pretrained(MODEL_ID)
        model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)
        model = model.to(device)
        print(f"Model loaded on device: {next(model.parameters()).device}")
    return processor, model

def extract_audio_from_video(video_path: str) -> Optional[str]:
    """Extract audio from video file and save as temporary WAV file"""
    try:
        print(f"Attempting to extract audio from: {video_path}")
        print(f"File exists: {os.path.exists(video_path)}")
        print(f"File size: {os.path.getsize(video_path) if os.path.exists(video_path) else 'N/A'} bytes")
        
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
        
        print(f"Audio extraction completed. Output file size: {os.path.getsize(temp_audio_path)} bytes")
        return temp_audio_path
    except Exception as e:
        print(f"Error extracting audio: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return None

def transcribe_audio(audio_path: str) -> Optional[str]:
    """Transcribe audio using Whisper - handles full length audio by chunking"""
    try:
        import librosa
        
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
            
            # Process chunk
            inputs = processor(chunk, sampling_rate=16000, return_tensors="pt")
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            # Generate transcription
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=224,
                    language="en",
                    task="transcribe"
                )
            
            chunk_transcription = processor.batch_decode(outputs, skip_special_tokens=True)[0]
            transcriptions.append(chunk_transcription)
        
        full_transcription = " ".join(transcriptions)
        print(f"Transcribed {len(transcriptions)} chunks, total length: {len(full_transcription)} characters")
        
        return full_transcription
    except Exception as e:
        print(f"Error during transcription: {e}")
        return None

def text_to_speech(text: str, output_path: str) -> bool:
    """Convert text to speech using gTTS"""
    try:
        tts = gTTS(text=text, lang='en')
        tts.save(output_path)
        return True
    except Exception as e:
        print(f"Error during text-to-speech conversion: {e}")
        return False

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Video Audio Text Service is running", "device": str(device)}

@app.post("/process-video/")
async def process_video(file: UploadFile = File(...)):
    """Process uploaded video file and return transcription and speech"""
    
    # Validate file type - check both content type and file extension
    valid_extensions = ['.mp4', '.mov', '.avi', '.mkv', '.wmv', '.flv', '.webm', '.m4v']
    file_extension = Path(file.filename).suffix.lower()
    
    if not (file.content_type and file.content_type.startswith('video/')) and file_extension not in valid_extensions:
        raise HTTPException(status_code=400, detail=f"File must be a video. Supported formats: {', '.join(valid_extensions)}")
    
    # Generate unique ID for this processing job
    job_id = uuid.uuid4().hex
    
    # Create temporary file for uploaded video
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{job_id}.{file.filename.split('.')[-1]}") as tmp_video:
            shutil.copyfileobj(file.file, tmp_video)
            video_path = tmp_video.name
        
        print(f"Video uploaded to: {video_path}")
        print(f"Uploaded file size: {os.path.getsize(video_path)} bytes")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {str(e)}")
    
    try:
        # Get base filename
        video_basename = Path(file.filename).stem
        
        # Extract audio
        print(f"Extracting audio from {file.filename}...")
        audio_path = extract_audio_from_video(video_path)
        if not audio_path:
            raise HTTPException(status_code=500, detail="Failed to extract audio")
        
        # Transcribe audio
        print("Transcribing audio...")
        transcription = transcribe_audio(audio_path)
        if not transcription:
            raise HTTPException(status_code=500, detail="Failed to transcribe audio")
        
        # Save transcription
        transcript_filename = f"{video_basename}_{job_id}_transcript.txt"
        transcript_path = TRANSCRIPTS_DIR / transcript_filename
        with open(transcript_path, 'w') as f:
            f.write(transcription)
        
        # Generate speech
        print("Converting to speech...")
        speech_filename = f"{video_basename}_{job_id}_speech.mp3"
        speech_path = SPEECH_DIR / speech_filename
        success = text_to_speech(transcription, str(speech_path))
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to generate speech")
        
        return {
            "job_id": job_id,
            "filename": file.filename,
            "transcription": transcription,
            "transcript_file": transcript_filename,
            "speech_file": speech_filename,
            "download_urls": {
                "transcript": f"/download/transcript/{transcript_filename}",
                "speech": f"/download/speech/{speech_filename}"
            }
        }
    
    finally:
        # Clean up temporary files
        if os.path.exists(video_path):
            os.remove(video_path)
        if 'audio_path' in locals() and audio_path and os.path.exists(audio_path):
            os.remove(audio_path)

@app.get("/download/transcript/{filename}")
async def download_transcript(filename: str):
    """Download transcript file"""
    file_path = TRANSCRIPTS_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Transcript file not found")
    return FileResponse(file_path, media_type='text/plain', filename=filename)

@app.get("/download/speech/{filename}")
async def download_speech(filename: str):
    """Download speech file"""
    file_path = SPEECH_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Speech file not found")
    return FileResponse(file_path, media_type='audio/mpeg', filename=filename)

@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get files for a specific job"""
    transcript_files = list(TRANSCRIPTS_DIR.glob(f"*{job_id}*"))
    speech_files = list(SPEECH_DIR.glob(f"*{job_id}*"))
    
    if not transcript_files and not speech_files:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return {
        "job_id": job_id,
        "transcript_files": [f.name for f in transcript_files],
        "speech_files": [f.name for f in speech_files]
    }

@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete all files for a specific job"""
    transcript_files = list(TRANSCRIPTS_DIR.glob(f"*{job_id}*"))
    speech_files = list(SPEECH_DIR.glob(f"*{job_id}*"))
    
    deleted_files = []
    for file_path in transcript_files + speech_files:
        file_path.unlink()
        deleted_files.append(file_path.name)
    
    return {"deleted_files": deleted_files}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
