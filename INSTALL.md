# 🚀 Quick Installation Guide

## One-Command Setup

```bash
# Install all dependencies
pip install -r requirements.txt
```

## Verify Installation

```bash
# Test that all dependencies are working
python verify_installation.py
```

## What's Included

The unified `requirements.txt` includes everything you need:

- ✅ **Web Framework**: FastAPI + Uvicorn
- ✅ **Video Processing**: MoviePy, PyTorch, Transformers
- ✅ **Multi-Platform Download**: yt-dlp (1000+ platforms)
- ✅ **Text-to-Speech**: Microsoft Edge-TTS
- ✅ **Database**: SQLAlchemy + async support
- ✅ **Authentication**: JWT + Google OAuth
- ✅ **Python 3.13 Compatibility**: Updated audio modules

## Optional Components

Some features require additional setup:

### MLX (Apple Silicon Optimization)
```bash
# For Apple Silicon users who want MLX acceleration
pip install mlx mlx-lm
```

### Google AI (Enhanced Summarization)
```bash
# For Google Generative AI features
pip install google-generativeai
```

## Start the Service

```bash
python async_video_service.py
```

Open http://localhost:8000 in your browser!

---

**That's it! One requirements file, one command, ready to go! 🎉**
