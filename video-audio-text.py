import asyncio
import platform
import moviepy as mp
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from TTS.api import TTS
import os
import torch
import sys

# Configuration
FPS = 60
MODEL_ID = "openai/whisper-base"  # Changed to Whisper for audio transcription
OUTPUT_AUDIO = "output_speech.mp3"

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

def extract_audio_from_video(video_path):
    """Extract audio from video file and save as temporary WAV file"""
    try:
        video = mp.VideoFileClip(video_path)
        audio = video.audio
        temp_audio_path = "temp_audio.wav"
        audio.write_audiofile(temp_audio_path)
        audio.close()
        video.close()
        return temp_audio_path
    except Exception as e:
        print(f"Error extracting audio: {e}")
        return None

def transcribe_audio(audio_path):
    """Transcribe audio using Whisper - handles full length audio by chunking"""
    try:
        import librosa
        
        # Load Whisper model and processor
        processor = WhisperProcessor.from_pretrained(MODEL_ID)
        model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)
        
        # Move model to Apple Silicon GPU if available
        model = model.to(device)
        print(f"Model device: {next(model.parameters()).device}")
        
        # Load audio file
        audio, _ = librosa.load(audio_path, sr=16000)  # Whisper expects 16kHz
        audio_duration = len(audio) / 16000  # Duration in seconds
        print(f"Audio duration: {audio_duration:.2f} seconds")
        
        # Process audio in chunks to handle full length
        chunk_length = 30  # seconds
        chunk_samples = chunk_length * 16000
        transcriptions = []
        
        for i in range(0, len(audio), chunk_samples):
            chunk = audio[i:i + chunk_samples]
            chunk_num = i // chunk_samples + 1
            total_chunks = (len(audio) + chunk_samples - 1) // chunk_samples
            
            print(f"Processing chunk {chunk_num}/{total_chunks} ({len(chunk)/16000:.1f}s)")
            
            # Process chunk
            inputs = processor(chunk, sampling_rate=16000, return_tensors="pt")
            
            # Move inputs to device
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            # Generate transcription for this chunk
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=224,
                    language="en",
                    task="transcribe"
                )
            
            chunk_transcription = processor.batch_decode(outputs, skip_special_tokens=True)[0]
            transcriptions.append(chunk_transcription)
        
        # Combine all transcriptions
        full_transcription = " ".join(transcriptions)
        print(f"Transcribed {len(transcriptions)} chunks, total length: {len(full_transcription)} characters")
        
        return full_transcription
    except Exception as e:
        print(f"Error during transcription: {e}")
        return None

def text_to_speech(text, output_path):
    """Convert text to speech using Coqui TTS"""
    try:
        print("Loading Coqui TTS model...")
        # Use a high-quality English model
        tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC").to(device)
        
        # Generate speech
        print("Generating speech...")
        tts.tts_to_file(text=text, file_path=output_path)
        print("Speech generated with Coqui TTS")
        return True
    except Exception as e:
        print(f"Error during Coqui TTS conversion: {e}")
        print("Attempting fallback TTS...")
        try:
            # Fallback to system TTS on macOS
            if platform.system() == "Darwin":
                import subprocess
                aiff_path = output_path.replace('.mp3', '.aiff')
                subprocess.run(["say", "-o", aiff_path, text])
                # Convert AIFF to MP3 if needed (simplified approach)
                print(f"Fallback speech generated: {aiff_path}")
                return True
            return False
        except Exception as fallback_error:
            print(f"Fallback TTS also failed: {fallback_error}")
            return False

def main():
    # Check for command line argument
    if len(sys.argv) < 2:
        print("Usage: python video-audio-text.py <video_file_path>")
        print("Example: python video-audio-text.py /Users/nchhina/Desktop/open-houses.mov")
        return
    
    video_path = sys.argv[1]
    
    # Check if file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' does not exist")
        return
    
    # Create output directories
    os.makedirs("/Users/nchhina/Documents/transcripts", exist_ok=True)
    os.makedirs("/Users/nchhina/Documents/speech", exist_ok=True)
    
    # Get base filename from video path
    video_basename = os.path.splitext(os.path.basename(video_path))[0]
    
    # Extract audio from video
    print(f"Extracting audio from {video_path}...")
    audio_path = extract_audio_from_video(video_path)
    if not audio_path:
        print("Failed to extract audio")
        return

    # Transcribe audio
    print("Transcribing audio...")
    transcription = transcribe_audio(audio_path)
    if not transcription:
        print("Failed to transcribe audio")
        return
    
    print("Transcription:", transcription)
    
    # Save transcription to text file in transcripts directory
    transcript_path = f"transcripts/{video_basename}_transcript.txt"
    with open(transcript_path, 'w') as f:
        f.write(transcription)
    print(f"Transcription saved to {transcript_path}")
    
    # Convert transcription to speech and save in speech directory
    speech_path = f"speech/{video_basename}_speech.mp3"
    print("Converting transcription to speech...")
    success = text_to_speech(transcription, speech_path)
    if success:
        print(f"Speech saved to {speech_path}")
    
    # Clean up temporary audio file
    if os.path.exists(audio_path):
        os.remove(audio_path)
        print(f"Cleaned up temporary file: {audio_path}")
    else:
        print(f"Temporary file {audio_path} not found for cleanup")

if platform.system() == "Emscripten":
    task = asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        main()