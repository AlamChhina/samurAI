# Video Audio Text Service

A comprehensive FastAPI service that processes videos to extract audio, generate transcriptions using AI, and convert text back to speech. Features user authentication, job tracking, Docker support, and a modern web interface.

## 🌟 Features

### Core Processing
- 🎥 **Video Processing**: Support for MP4, MOV, AVI, MKV, WebM and more
- 🎯 **AI Transcription**: Powered by OpenAI Whisper for accurate speech-to-text
- 🔊 **Premium Text-to-Speech**: Microsoft Edge-TTS Neural Voices (free & high-quality)
- 📄 **Document Processing**: PDF and text file summarization and speech conversion
- 🎬 **YouTube Integration**: Direct URL processing with yt-dlp
- 🌐 **Multi-Platform Support**: YouTube, Vimeo, and other video platforms

### User Experience
- 👤 **Google OAuth Authentication**: Secure user management
- 📊 **Modern Dashboard**: Beautiful web interface with job management
- 📋 **Copy Functionality**: One-click summary copying with visual feedback
- 🔗 **Clickable Links**: Job titles link directly to source URLs
- 📈 **Usage Tracking**: Monitor processing limits and usage statistics
- ⚡ **Real-time Updates**: Live job status and progress tracking

### Technical
- 🚀 **Async Processing**: Non-blocking background job processing
- 💾 **SQLite Database**: Persistent storage for users, jobs, and preferences
- 🐳 **Docker Support**: Complete containerization with Docker Compose
- 🍎 **Apple Silicon**: Optimized for Apple MPS acceleration
- 🔧 **Health Monitoring**: Built-in health checks and monitoring
- 📦 **Easy Deployment**: Production-ready with resource management

## 🎤 Text-to-Speech Engine

### Edge-TTS Neural Voices
- **Engine**: Microsoft Edge-TTS (completely free)
- **Quality**: Premium neural voices indistinguishable from human speech
- **Languages**: 20+ voices in multiple languages and accents
- **Performance**: Fast local processing, no API costs or limits

### Recommended Voices by Use Case
- 📚 **Audiobooks/Long content**: `en-US-JennyMultilingualNeural`
- 🎥 **Video narration**: `en-US-AriaNeural` 
- 📢 **Announcements**: `en-US-DavisNeural` (male)
- 💼 **Professional**: `en-US-MonicaNeural`
- 🎧 **Podcasts**: `en-US-SaraNeural`

## 🚀 Quick Start

### Option 1: Docker (Recommended)

1. **Setup Environment**:
   ```bash
   # Copy environment template
   cp .env.example .env
   
   # Edit .env with your Google OAuth credentials
   nano .env
   ```

2. **Start with Docker Compose**:
   ```bash
   # Build and start the service
   docker-compose up --build
   
   # Or run in background
   docker-compose up -d --build
   ```

3. **Access the Service**:
   - **Main Application**: http://localhost:8000
   - **Dashboard**: http://localhost:8000/dashboard
   - **API Documentation**: http://localhost:8000/docs
   - **Health Check**: http://localhost:8000/health

### Option 2: Local Development

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your credentials
   ```

3. **Start the Service**:
   ```bash
   python async_video_service.py
   # Or use the startup script
   ./start_async_service.sh
   ```

## 🔧 Configuration

### Environment Variables (.env)
```bash
# Google OAuth (Required)
GOOGLE_CLIENT_ID=your_google_client_id
GOOGLE_CLIENT_SECRET=your_google_client_secret
JWT_SECRET_KEY=your_jwt_secret

# Service Configuration
HOST=0.0.0.0
PORT=8000
DEBUG=false

# Database
DATABASE_URL=sqlite:///./data/video_audio_service.db

# TTS Configuration
PREFERRED_VOICE=en-US-AriaNeural

# Stripe (Optional for payments)
STRIPE_PUBLISHABLE_KEY=your_stripe_key
STRIPE_SECRET_KEY=your_stripe_secret
STRIPE_ENDPOINT_SECRET=your_stripe_webhook_secret
```

### Google OAuth Setup
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing
3. Enable Google+ API
4. Create OAuth 2.0 credentials
5. Add authorized redirect URI: `http://localhost:8000/auth/callback`

## 📡 API Endpoints

### Authentication
- `GET /` - Home page
- `GET /auth/google` - Google OAuth login
- `GET /auth/callback` - OAuth callback
- `GET /auth/logout` - Logout
- `GET /dashboard` - User dashboard

### Processing
- `POST /api/process-video/` - Upload and process video
- `POST /api/process-url/` - Process video from URL
- `POST /api/process-transcript/` - Process text/PDF files
- `GET /api/jobs/` - Get user's jobs
- `GET /api/jobs/{job_id}` - Get specific job details
- `DELETE /api/jobs/{job_id}` - Delete job

### Files & Downloads
- `GET /download/transcript/{filename}` - Download transcript
- `GET /download/speech/{filename}` - Download speech file
- `GET /download/original-audio/{filename}` - Download original audio

### System
- `GET /health` - Health check endpoint
- `GET /api/usage-stats/` - User usage statistics

## 🐳 Docker Deployment

### Docker Compose Features
- **Multi-stage builds** for optimized images
- **Volume persistence** for database and outputs
- **Health checks** with automatic recovery
- **Resource limits** and management
- **Security hardening** with non-root user

### Volume Mounts
- `./data:/app/data` - Database and app data persistence
- `./docker-outputs:/app/outputs` - Generated files
- `./docker-logs:/app/logs` - Application logs
- `./.env:/app/.env:ro` - Environment configuration

### Docker Commands
```bash
# Build custom image
docker build -t video-audio-service .

# Run with compose (recommended)
docker-compose up -d --build

# View logs
docker-compose logs -f

# Stop service
docker-compose down

# Clean restart (removes data)
docker-compose down -v && docker-compose up --build
```

## 📁 File Structure
```
├── async_video_service.py     # Main FastAPI application
├── database.py               # SQLAlchemy models and database functions
├── edge_tts_helper.py        # Voice configuration and testing
├── docker-compose.yml        # Docker Compose configuration
├── Dockerfile               # Multi-stage Docker build
├── .dockerignore           # Docker build optimization
├── requirements.txt        # Python dependencies
├── .env.example           # Environment template
├── templates/             # Jinja2 HTML templates
│   ├── index.html        # Home page
│   ├── dashboard.html    # User dashboard
│   └── profile.html      # User profile
├── static/               # CSS and JavaScript
│   └── style.css        # Application styles
├── data/                # Persistent database storage
├── outputs/             # Generated files
│   ├── transcripts/     # Text transcriptions
│   ├── speech/         # Generated speech files
│   └── summaries/      # AI-generated summaries
└── docker-outputs/     # Docker volume mount for outputs
```

## 🧪 Testing & Development

### Voice Testing
```bash
# Test available voices
python edge_tts_helper.py

# Test voice quality
python test_voice_quality.py
```

### Service Testing
```bash
# Test async processing
python test_async_service.py

# Test video platforms
python test_video_platforms.py
```

### Health Checks
```bash
# Check service health
curl http://localhost:8000/health

# Check Docker container health
docker-compose ps
```

## 🔧 Advanced Features

### Usage Tracking & Limits
- Built-in usage tracking per user
- Configurable processing limits
- Real-time usage statistics
- Subscription management support

### Error Handling
- Comprehensive error handling for video platforms
- Fallback download configurations
- Graceful degradation for unsupported formats
- User-friendly error messages

### Performance Optimization
- Async processing with background tasks
- Efficient video processing with FFmpeg
- Chunked text processing for long content
- Resource-aware processing limits

### Security Features
- Google OAuth integration
- JWT token authentication
- Non-root Docker containers
- Input validation and sanitization

## 📈 Benefits

### ✅ **Advantages**
- **Cost Effective**: Edge-TTS is completely free
- **High Quality**: Premium neural voices
- **Privacy**: All processing done locally
- **Reliable**: No API rate limits or outages
- **Fast**: Local processing, no network delays
- **Scalable**: Docker deployment ready
- **Maintainable**: Clean, modern codebase

### 🎯 **Use Cases**
- Professional presentations
- Audiobook creation
- Video voiceovers
- Podcast generation
- Educational content
- Document narration
- Accessibility tools

## 🛠 Troubleshooting

### Common Issues

**Docker Build Fails**:
```bash
# Clean Docker cache
docker system prune -a
docker-compose down -v
```

**Package Installation Issues**:
```bash
# Update pip and reinstall
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt --no-cache-dir
```

**Google OAuth Issues**:
- Verify redirect URI in Google Cloud Console
- Check `.env` file credentials
- Ensure Google+ API is enabled

**Video Processing Failures**:
- Check video format compatibility
- Verify FFmpeg installation
- Review error logs in dashboard

### Performance Tuning

**For Apple Silicon**:
```bash
# Install optimized PyTorch
pip install torch torchvision torchaudio
```

**For Production**:
- Use Docker deployment
- Configure resource limits
- Enable health monitoring
- Set up log rotation

## 📄 License & Support

This project is designed for educational and development purposes. For production deployment, ensure compliance with all service terms and conditions.

For issues and feature requests, please check the application logs and health endpoints for diagnostic information.

---

**Version**: 2.0  
**Last Updated**: July 2025  
**Docker Support**: ✅  
**Apple Silicon**: ✅  
**Production Ready**: ✅
