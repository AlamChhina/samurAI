# Simplified Video Audio Text Service

## Overview
This service has been simplified to use only **Edge-TTS (Microsoft Neural Voices)** for high-quality, natural text-to-speech generation. All other TTS engines have been removed for a cleaner, more focused codebase.

## Features
✅ **Video Upload & Processing**
✅ **Audio Extraction** from video files
✅ **Speech Transcription** using Whisper AI
✅ **High-Quality Text-to-Speech** using Edge-TTS
✅ **Google OAuth Authentication**
✅ **Job Management** with database storage
✅ **File Download** (transcripts and speech)

## TTS Engine: Edge-TTS Only
- **Engine**: Microsoft Edge-TTS Neural Voices
- **Quality**: High-quality, natural-sounding speech
- **Cost**: Completely free
- **Voices**: 20+ premium neural voices available
- **Language**: English (US, UK, AU, CA, etc.)

## Configuration

### Current Voice
```bash
PREFERRED_VOICE=en-US-AriaNeural
```

### Recommended Voices by Use Case
- **📚 Audiobooks/Long content**: `en-US-JennyMultilingualNeural`
- **🎥 Video narration**: `en-US-AriaNeural`
- **📢 Announcements**: `en-US-DavisNeural` (male)
- **💼 Professional**: `en-US-MonicaNeural`
- **🎧 Podcasts**: `en-US-SaraNeural`

### Voice Configuration Helper
```bash
python edge_tts_helper.py
```
This helper shows all available voices and lets you test the current configuration.

## API Endpoints

### Authentication
- `GET /` - Home page
- `GET /auth/google` - Google OAuth login
- `GET /auth/callback` - OAuth callback
- `GET /auth/logout` - Logout
- `GET /dashboard` - User dashboard

### Video Processing
- `POST /api/process-video/` - Upload and process video
- `GET /api/jobs/` - Get user's jobs
- `GET /api/jobs/{job_id}` - Get specific job
- `DELETE /api/jobs/{job_id}` - Delete job

### File Downloads
- `GET /download/transcript/{filename}` - Download transcript
- `GET /download/speech/{filename}` - Download speech file

## Dependencies
- `fastapi` - Web framework
- `edge-tts` - Text-to-speech engine
- `transformers` - Whisper AI for transcription
- `moviepy` - Video processing
- `sqlalchemy` - Database ORM
- `httpx` - HTTP client for OAuth
- `python-dotenv` - Environment variables

## Quick Start

1. **Start the service**:
   ```bash
   python async_video_service.py
   ```

2. **Open in browser**:
   ```
   http://localhost:8000
   ```

3. **Test TTS voices**:
   ```bash
   python edge_tts_helper.py
   ```

4. **Test voice quality**:
   ```bash
   python test_voice_quality.py
   ```

## File Structure
```
async_video_service.py     # Main service
edge_tts_helper.py        # Voice configuration helper
test_voice_quality.py     # Voice quality testing
.env                      # Configuration
requirements.txt          # All dependencies
database.py              # Database models
templates/               # HTML templates
static/                  # CSS/JS files
outputs/                 # Generated files
  transcripts/           # Text transcriptions
  speech/               # Generated speech files
```

## Benefits of Simplified Approach

### ✅ **Pros**
- **Simpler codebase** - easier to maintain
- **No external API costs** - Edge-TTS is free
- **High quality voices** - Microsoft Neural voices sound natural
- **Fast processing** - no API network delays
- **Reliable** - no API rate limits or outages
- **Privacy** - all processing done locally

### 🔧 **Configuration**
- Single voice setting in `.env`
- Easy to change voices
- Voice testing helper included

## Voice Quality
The Edge-TTS neural voices provide **excellent quality** that's suitable for:
- Professional presentations
- Audiobook narration
- Video voiceovers
- Podcast generation
- Educational content

The voices are virtually indistinguishable from human speech for most use cases.
