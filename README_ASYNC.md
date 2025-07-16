# Video Audio Text Service with User Authentication

A robust async FastAPI service that processes videos to extract audio, generate transcriptions using AI, and convert text back to speech. Features user authentication via Google OAuth, job tracking, and a beautiful dashboard.

## Features

- 🎥 **Video Processing**: Support for MP4, MOV, AVI, MKV, WebM and more
- 🎯 **AI Transcription**: Powered by OpenAI Whisper for accurate transcripts
- 🔊 **Text-to-Speech**: Convert transcripts back to natural speech using gTTS
- 👤 **User Authentication**: Google OAuth integration for user management
- 📊 **Dashboard**: Beautiful web interface to manage jobs and downloads
- 🚀 **Async Processing**: Non-blocking background job processing
- 💾 **SQLite Database**: Track users, jobs, and processing status
- 🍎 **Apple Silicon**: Optimized for Apple MPS acceleration

## Installation

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements.txt
   ```

2. **Set up Google OAuth:**
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Create a new project or select existing
   - Enable Google+ API
   - Create OAuth 2.0 credentials
   - Add authorized redirect URI: `http://localhost:8000/auth/callback`

3. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your Google OAuth credentials
   ```

## Usage

### Start the Service

```bash
# Start the async service with authentication
./start_async_service.sh

# Or directly
python async_video_service.py
```

### Access the Service

- **Home Page**: http://localhost:8000/
- **Dashboard**: http://localhost:8000/dashboard
- **API Docs**: http://localhost:8000/docs
- **Interactive API**: http://localhost:8000/redoc

### CLI Usage

You can still use the CLI script directly:

```bash
python video-audio-text.py /path/to/your/video.mp4
```

## API Endpoints

### Authentication
- `GET /auth/google` - Redirect to Google OAuth
- `GET /auth/callback` - OAuth callback handler

### Jobs Management
- `POST /api/process-video/` - Upload and process video
- `GET /api/jobs/` - Get user's jobs
- `GET /api/jobs/{job_id}` - Get specific job details
- `DELETE /api/jobs/{job_id}` - Delete job and files

### File Downloads
- `GET /download/transcript/{filename}` - Download transcript
- `GET /download/speech/{filename}` - Download speech file

## File Structure

```
python-scripts/
├── async_video_service.py      # Main async FastAPI service
├── database.py                 # SQLite models
├── video-audio-text.py         # CLI script
├── templates/                  # HTML templates
│   ├── index.html             # Home page
│   └── dashboard.html         # User dashboard
├── static/                    # CSS and assets
├── outputs/                   # Generated files
│   ├── transcripts/           # Text transcriptions
│   └── speech/               # Generated speech
├── requirements.txt           # All service dependencies
└── .env.example              # Environment template
```

## Database Schema

### Users Table
- `id` - UUID primary key
- `google_id` - Google OAuth ID
- `email` - User email
- `name` - User name
- `picture` - Profile picture URL
- `created_at` - Account creation time

### Jobs Table
- `id` - UUID primary key
- `user_id` - Foreign key to users
- `filename` - Original video filename
- `status` - pending, processing, completed, failed
- `transcription` - Generated text
- `transcript_file` - Transcript filename
- `speech_file` - Speech filename
- `error_message` - Error details if failed
- `created_at` - Job creation time
- `completed_at` - Job completion time

## Technology Stack

- **Backend**: FastAPI (async)
- **Database**: SQLite with SQLAlchemy
- **AI/ML**: OpenAI Whisper, gTTS
- **Video Processing**: MoviePy
- **Authentication**: Google OAuth 2.0
- **Frontend**: HTML/CSS/JavaScript with Tailwind CSS
- **File Handling**: aiofiles for async I/O

## Testing

```bash
# Test the service
python test_async_service.py

# Test CLI functionality
python video-audio-text.py /path/to/test/video.mp4
```

## Environment Variables

Create a `.env` file with:

```env
GOOGLE_CLIENT_ID=your-google-client-id
GOOGLE_CLIENT_SECRET=your-google-client-secret
JWT_SECRET_KEY=your-secure-random-key
DATABASE_URL=sqlite:///./video_audio_service.db
HOST=localhost
PORT=8000
DEBUG=true
```

## Production Deployment

1. **Security**: Use proper JWT tokens with expiration
2. **Database**: Consider PostgreSQL for production
3. **Storage**: Use cloud storage for video/audio files
4. **Scaling**: Add Redis for job queues
5. **Monitoring**: Add logging and error tracking

## Troubleshooting

### Common Issues

1. **FFmpeg not found**: Install FFmpeg for video processing
2. **Google OAuth errors**: Check redirect URI configuration
3. **Audio extraction fails**: Ensure video has audio track
4. **MPS errors**: Fallback to CPU if Apple Silicon issues

### Debug Mode

Set `DEBUG=true` in `.env` for detailed error messages.

## License

MIT License - feel free to modify and distribute.

## Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Submit pull request

---

Built with ❤️ for seamless video-to-text processing with user management.
