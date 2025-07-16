#!/usr/bin/env python3
"""
Installation verification script for Video Audio Text Service
Tests that all required dependencies can be imported successfully
"""

import sys
import importlib

def test_imports():
    """Test importing all required dependencies"""
    
    print("🔍 Testing Video Audio Text Service Dependencies\n")
    
    # Core dependencies
    dependencies = {
        # Web Framework
        "fastapi": "FastAPI web framework",
        "uvicorn": "ASGI server",
        "jinja2": "Template engine",
        
        # Video/Audio Processing
        "moviepy": "Video processing",
        "torch": "PyTorch ML framework", 
        "transformers": "Hugging Face transformers",
        "librosa": "Audio analysis",
        
        # Video Download
        "yt_dlp": "Video download (multi-platform)",
        "youtube_transcript_api": "YouTube transcript API",
        
        # Text-to-Speech
        "edge_tts": "Microsoft Edge TTS",
        
        # Database & Async
        "sqlalchemy": "Database ORM",
        "aiofiles": "Async file operations",
        "httpx": "Async HTTP client",
        
        # Authentication
        "jwt": "JSON Web Tokens (PyJWT)",
        
        # Web Scraping
        "requests": "HTTP library",
        "bs4": "Beautiful Soup (HTML parsing)",
        "markdown": "Markdown processing",
    }
    
    # Optional dependencies
    optional_deps = {
        "google.generativeai": "Google Generative AI",
        "accelerate": "Hugging Face Accelerate",
    }
    
    # Test core dependencies
    failed_core = []
    for module, description in dependencies.items():
        try:
            importlib.import_module(module)
            print(f"✅ {module:<25} - {description}")
        except ImportError as e:
            print(f"❌ {module:<25} - {description} (FAILED: {e})")
            failed_core.append(module)
    
    print("\n" + "="*60)
    print("Optional Dependencies:")
    print("="*60)
    
    # Test optional dependencies
    failed_optional = []
    for module, description in optional_deps.items():
        try:
            importlib.import_module(module)
            print(f"✅ {module:<25} - {description}")
        except ImportError:
            print(f"⚠️  {module:<25} - {description} (Optional - not installed)")
            failed_optional.append(module)
    
    print("\n" + "="*60)
    print("Summary:")
    print("="*60)
    
    if not failed_core:
        print("🎉 All core dependencies are installed correctly!")
        print("🚀 Ready to run the Video Audio Text Service!")
    else:
        print(f"❌ {len(failed_core)} core dependencies failed to import:")
        for dep in failed_core:
            print(f"   - {dep}")
        print("\n💡 Run: pip install -r requirements.txt")
        return False
    
    if failed_optional:
        print(f"\nℹ️  {len(failed_optional)} optional dependencies not installed (this is OK)")
    
    print("\n🔧 Installation Command:")
    print("   pip install -r requirements.txt")
    
    return True

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
