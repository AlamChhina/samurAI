import requests
import json

# Test the FastAPI service
BASE_URL = "http://localhost:8000"

def test_service():
    # Health check
    print("🏥 Testing health check...")
    response = requests.get(f"{BASE_URL}/")
    print(f"Health check: {response.json()}")
    
    # Test file upload (you'll need to provide a video file)
    print("\n📤 To test video upload:")
    print("Use this curl command or the web interface at http://localhost:8000/docs")
    print(f"curl -X POST '{BASE_URL}/process-video/' -F 'file=@/path/to/your/video.mp4'")
    
    print("\n🌐 Web interface available at:")
    print(f"📖 API Docs: {BASE_URL}/docs")
    print(f"🔧 Interactive API: {BASE_URL}/redoc")

if __name__ == "__main__":
    try:
        test_service()
    except requests.exceptions.ConnectionError:
        print("❌ Service not running. Start it with: python video_audio_service.py")
