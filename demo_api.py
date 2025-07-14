"""
Demo script showing how to interact with the Video Audio Text Service API
This demonstrates programmatic usage of the service endpoints
"""

import requests
import json
import time
from pathlib import Path

BASE_URL = "http://localhost:8000"

def demo_api_usage():
    """Demo how to use the API programmatically"""
    print("🚀 Video Audio Text Service - API Demo")
    print("=" * 50)
    
    # Note: For the demo, we'll show the API structure
    # In real usage, you'd need to authenticate first
    
    print("\n1. 📊 Getting User Jobs:")
    print("   GET /api/jobs/")
    print("   Headers: {'Authorization': 'Bearer <user_token>'}")
    print("   Returns: List of user's processing jobs")
    
    print("\n2. 📤 Uploading Video:")
    print("   POST /api/process-video/")
    print("   Files: {'file': <video_file>}")
    print("   Headers: {'Authorization': 'Bearer <user_token>'}")
    print("   Returns: {'job_id': '<uuid>', 'status': 'pending'}")
    
    print("\n3. 🔍 Checking Job Status:")
    print("   GET /api/jobs/{job_id}")
    print("   Headers: {'Authorization': 'Bearer <user_token>'}")
    print("   Returns: Job details with current status")
    
    print("\n4. 📥 Downloading Results:")
    print("   GET /download/transcript/{filename}")
    print("   GET /download/speech/{filename}")
    print("   Headers: {'Authorization': 'Bearer <user_token>'}")
    print("   Returns: File download")
    
    print("\n5. 🗑️  Deleting Job:")
    print("   DELETE /api/jobs/{job_id}")
    print("   Headers: {'Authorization': 'Bearer <user_token>'}")
    print("   Returns: Confirmation message")
    
    print("\n" + "=" * 50)
    print("📋 Example Usage Flow:")
    print("1. User authenticates via Google OAuth")
    print("2. Upload video file to /api/process-video/")
    print("3. Poll /api/jobs/{job_id} for completion")
    print("4. Download transcript and speech files")
    print("5. Optionally delete job when done")
    
    print("\n🔐 Authentication:")
    print("- Web UI: Login via Google OAuth at /auth/google")
    print("- API: Use Bearer token in Authorization header")
    print("- Token format: 'Bearer <user_email>' (simplified for demo)")
    
    print("\n📁 File Formats Supported:")
    print("- Video: MP4, MOV, AVI, MKV, WebM, WMV, FLV, M4V")
    print("- Output: TXT (transcripts), MP3 (speech)")
    
    print("\n🌐 Web Interface:")
    print(f"- Home: {BASE_URL}/")
    print(f"- Dashboard: {BASE_URL}/dashboard")
    print(f"- API Docs: {BASE_URL}/docs")

if __name__ == "__main__":
    demo_api_usage()
