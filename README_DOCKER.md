# Docker Setup for Video Audio Text Service

This guide shows how to run the Video Audio Text Service using Docker.

## Prerequisites

- Docker installed on your system
- Docker Compose (included with Docker Desktop)

## Quick Start

### 1. Clone and Setup Environment

```bash
# Clone the repository (if not already done)
git clone <your-repo-url>
cd python-scripts

# Copy environment template
cp .env.example .env

# Edit .env file with your credentials
nano .env  # or use your preferred editor
```

### 2. Build and Run with Docker Compose

```bash
# Build and start the service
docker-compose up --build

# Or run in detached mode (background)
docker-compose up -d --build
```

The service will be available at:
- **Main Application**: http://localhost:8000
- **Dashboard**: http://localhost:8000/dashboard
- **API Documentation**: http://localhost:8000/docs

### 3. Stop the Service

```bash
# Stop the service
docker-compose down

# Stop and remove volumes (clears data)
docker-compose down -v
```

## Manual Docker Build

If you prefer to build and run manually:

```bash
# Build the image
docker build -t video-audio-service .

# Run the container
docker run -d \
  --name video-audio-service \
  -p 8000:8000 \
  -v $(pwd)/docker-outputs:/app/outputs \
  -v $(pwd)/data:/app/data \
  --env-file .env \
  video-audio-service
```

## Environment Configuration

### Required Environment Variables

Edit your `.env` file with these values:

```bash
# Google OAuth Configuration
GOOGLE_CLIENT_ID=your-google-client-id-here
GOOGLE_CLIENT_SECRET=your-google-client-secret-here

# JWT Secret Key (generate a secure random string)
JWT_SECRET_KEY=your-secure-secret-key-here

# Database URL
DATABASE_URL=sqlite:///./video_audio_service.db

# Service Configuration
HOST=0.0.0.0
PORT=8000
DEBUG=false
```

### Optional Environment Variables

```bash
# Stripe Payment Processing (optional)
STRIPE_PUBLISHABLE_KEY=your-stripe-publishable-key
STRIPE_SECRET_KEY=your-stripe-secret-key
STRIPE_ENDPOINT_SECRET=your-stripe-webhook-secret
```

## Volume Mounts

The Docker setup includes several volume mounts for data persistence:

- `./data:/app/data` - **Database and app data** (includes SQLite database file)
- `./docker-outputs:/app/outputs` - Generated audio files and transcripts
- `./docker-logs:/app/logs` - Application logs (optional)
- `./.env:/app/.env:ro` - Environment configuration (read-only)

### Important Database Notes

The SQLite database is configured to be stored in the mounted `./data` directory:
- **Database path**: `./data/video_audio_service.db`
- **Persistence**: Survives container restarts and rebuilds
- **Backup**: Simply copy the `./data` directory

Make sure the `./data` directory exists before starting:
```bash
mkdir -p data docker-outputs docker-logs
```

## Troubleshooting

### Service Won't Start

1. **Check logs**:
   ```bash
   docker-compose logs -f
   ```

2. **Verify environment variables**:
   ```bash
   # Check if .env file exists and has correct values
   cat .env
   ```

3. **Check port conflicts**:
   ```bash
   # Make sure port 8000 isn't in use
   lsof -i :8000
   ```

### Permission Issues

If you encounter permission issues with volume mounts:

```bash
# Fix permissions for output directories
sudo chown -R $USER:$USER docker-outputs/
sudo chown -R $USER:$USER docker-logs/
sudo chown -R $USER:$USER data/
```

### Memory Issues

If the service runs out of memory:

1. **Increase Docker memory limits** in Docker Desktop settings
2. **Adjust resource limits** in docker-compose.yml:
   ```yaml
   deploy:
     resources:
       limits:
         memory: 8G  # Increase as needed
   ```

### FFmpeg Issues

If you encounter FFmpeg-related errors:

```bash
# Check if FFmpeg is available in the container
docker exec -it video-audio-service which ffmpeg
docker exec -it video-audio-service ffmpeg -version
```

## Health Check

The service includes a health check endpoint:

```bash
# Check service health
curl http://localhost:8000/health

# Or check via Docker
docker exec -it video-audio-service curl http://localhost:8000/health
```

## Development Mode

For development with live reload:

```bash
# Edit docker-compose.yml to enable debug mode
environment:
  - DEBUG=true

# Mount source code for live reload
volumes:
  - .:/app

# Rebuild and run
docker-compose up --build
```

## Production Deployment

For production deployment:

1. **Use environment-specific .env file**
2. **Set DEBUG=false**
3. **Configure proper secrets management**
4. **Set up reverse proxy (nginx)**
5. **Configure SSL certificates**
6. **Set up monitoring and logging**

Example production docker-compose.yml additions:

```yaml
services:
  video-audio-service:
    restart: always
    environment:
      - DEBUG=false
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

## Updating the Service

```bash
# Pull latest changes
git pull

# Rebuild and restart
docker-compose up --build -d

# Clean up old images (optional)
docker image prune
```
