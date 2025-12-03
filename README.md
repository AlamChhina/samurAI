# samurAI

samurAI is a FastAPI-based service that turns videos, URLs, and documents into useful audio and text.  
It extracts audio, runs AI transcription, generates concise summaries, and can convert text back into natural-sounding speech.  
The app includes user authentication, job tracking, a modern dashboard, and full Docker support for easy deployment.  

---

## Features

- **Multi-format input** – upload local video files or provide URLs from platforms like YouTube, Vimeo, and more
- **AI transcription** – uses OpenAI Whisper for accurate speech-to-text on long-form content
- **Text-to-speech** – converts summaries and transcripts back into audio using Microsoft Edge-TTS neural voices  
- **Document support** – process PDFs and plain text files for summarization and audio generation  
- **Job dashboard** – track the status of processing jobs, view details, and download outputs from a web UI  
- **Real-time updates** – live status for queued, running, and completed jobs  
- **Authentication** – Google OAuth login with JWT-based sessions for protected routes and usage tracking
- **Dockerized** – ready-to-run with Docker Compose, including optional GPU acceleration for Whisper  
- **Health & monitoring** – health check endpoints and basic monitoring hooks for production deployments  

---

## Authentication & Access

samurAI uses **Google OAuth** for login and JWT tokens for authenticated API access. To run it locally or in your own environment, you'll need:

1. A Google Cloud project with OAuth 2.0 credentials (Web application).
2. A configured `.env` file containing your Google client details and JWT secret.
3. A valid redirect URI pointing to your local or deployed samurAI instance (e.g. `http://localhost:8000/auth/callback`).

Typical environment variables (names can be adjusted to match your actual code):

```bash
# Google OAuth
GOOGLE_CLIENT_ID=your_google_client_id
GOOGLE_CLIENT_SECRET=your_google_client_secret
JWT_SECRET_KEY=your_jwt_secret

# Service configuration
HOST=0.0.0.0
PORT=8000
DEBUG=false

# Database
DATABASE_URL=sqlite:///./data/video_audio_service.db

# Text-to-speech
PREFERRED_VOICE=en-US-AriaNeural
```

---

## How It Works

### Sign in

- Open the app in your browser.
- Sign in with Google using the OAuth flow.
- You'll be redirected back to the main dashboard once authenticated.

### Create a job

- Upload a video file or provide a URL to a supported video platform.
- Optionally choose processing options (transcription only, summary + TTS, etc.).
- Submit the job; it will appear in your job list with a pending status.

### Processing

- samurAI extracts the audio, runs Whisper transcription, and generates summaries.
- If enabled, it then converts the text output into speech using Edge-TTS.
- Jobs are processed asynchronously so the UI stays responsive.

### Review results

- Open a job from the dashboard to see its transcript, summary, and links to any generated audio files.
- Download transcripts, audio, or other artifacts directly from the job details page.

---

## Tech Stack

- **Backend**: FastAPI, Python
- **AI / Processing**: OpenAI Whisper, FFmpeg, Edge-TTS neural voices
- **Auth**: Google OAuth 2.0, JWT
- **Database**: SQLite (with SQLAlchemy)
- **Frontend**: HTML templates + modern dashboard (Jinja2 + static assets)
- **Deployment**: Docker, Docker Compose (CPU and optional GPU variants)
- **Platform Support**: Apple Silicon, NVIDIA CUDA, and standard CPU-only environments

---

## Running Locally

### Prerequisites

- Python 3.10+
- FFmpeg installed on your system
- A Google Cloud project with OAuth credentials
- (Optional) Docker & Docker Compose for containerized deployment

### Environment setup

```bash
# Clone the repository
git clone https://github.com/AlamChhina/samurAI.git
cd samurAI

# Create environment file
cp .env.example .env
# Edit .env with your Google OAuth credentials and other settings
```

### Option 1: Local Python dev

```bash
# Install dependencies
pip install -r requirements.txt

# Run the service
python async_video_service.py   # or use your project's main entrypoint
```

Then open http://localhost:8000 in your browser to access the UI and http://localhost:8000/docs for the interactive API docs.

### Option 2: Docker (recommended)

```bash
# CPU-only
docker compose up --build

# Or run in the background
docker compose up -d --build
```

For GPU acceleration (if you have an NVIDIA GPU and drivers configured):

```bash
docker compose -f docker-compose.gpu.yml up --build
```

---

## Notes

- samurAI is a personal / educational project and is not affiliated with or endorsed by Google, OpenAI, or Microsoft.
- Before using in production, review service limits, API terms of use, and security best practices for your environment.
