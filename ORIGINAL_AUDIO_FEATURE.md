# Original Audio Feature Implementation

## Overview
Added support for keeping and serving the original audio from video sources (YouTube, Vimeo, etc.) alongside the generated TTS audio.

## Changes Made

### 1. Database Schema
- Added `original_audio_file` column to the `jobs` table
- Created migration script to update existing databases

### 2. API Models
- Updated `JobResponse` to include `original_audio_file` field
- Updated `YouTubeRequest`, `VideoUploadRequest`, and `TranscriptRequest` to include `keep_original_audio` boolean parameter

### 3. Core Functions
- **`download_original_audio_from_url()`**: New function to extract high-quality audio directly from video sources using yt-dlp
- **`process_youtube_background()`**: Updated to support original audio extraction when `keep_original_audio=True`

### 4. API Endpoints
- **`/download/original-audio/{filename}`**: New endpoint to serve original audio files
- Updated existing endpoints to pass the `keep_original_audio` parameter

### 5. File Management
- Original audio files are stored in the `SPEECH_DIR` directory
- Updated `delete_job()` function to clean up original audio files

## Usage

### For API Users
```python
# Request with original audio
request = {
    "url": "https://www.youtube.com/watch?v=example",
    "summary_prompt": "Custom summary prompt",
    "keep_original_audio": True  # New parameter
}

# Response will include original_audio_file if available
response = {
    "original_audio_file": "video_title_job-id_original_audio.mp3",
    # ... other fields
}
```

### For Web Users
The frontend can now offer users a choice between:
1. **Generated TTS Audio**: High-quality text-to-speech from transcript
2. **Original Audio**: The actual audio from the video source

## Benefits
1. **User Choice**: Users can choose between TTS or original audio
2. **Quality**: Original audio maintains the speaker's natural voice, intonation, and audio quality
3. **Efficiency**: For users who prefer original audio, this saves TTS processing time
4. **Backup**: Provides a fallback option if TTS fails

## File Structure
```
outputs/
├── speech/
│   ├── video_title_job-id_speech.mp3          # Generated TTS
│   ├── video_title_job-id_original_audio.mp3  # Original audio (NEW)
│   └── video_title_job-id_summary_speech.mp3  # Summary TTS
├── transcripts/
│   └── video_title_job-id_transcript.txt
└── summaries/
    └── video_title_job-id_summary.txt
```

## Technical Details
- Uses yt-dlp with `bestaudio` format for highest quality
- Extracts to MP3 format at 192kbps quality
- Properly handles file cleanup and error scenarios
- Maintains backward compatibility with existing API clients

## Testing
Run the test script to verify functionality:
```bash
python test_original_audio.py
```

This feature provides users with more flexibility in how they consume audio content from processed videos.
