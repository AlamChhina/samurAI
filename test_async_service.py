import requests
import json
import time

# Test the Async FastAPI service
BASE_URL = "http://localhost:8000"

def test_async_service():
    print("🚀 Testing Async Video Audio Text Service")
    print("=" * 50)
    
    # Health check - Home page
    print("\n🏠 Testing home page...")
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"✅ Home page: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("❌ Service not running. Start it with: ./start_async_service.sh")
        return
    
    # Test API documentation
    print("\n📖 Testing API documentation...")
    try:
        response = requests.get(f"{BASE_URL}/docs")
        print(f"✅ API docs: {response.status_code}")
    except:
        print("⚠️  API docs endpoint error")
    
    # Test Google OAuth redirect (should redirect)
    print("\n🔐 Testing Google OAuth...")
    try:
        response = requests.get(f"{BASE_URL}/auth/google", allow_redirects=False)
        if response.status_code in [302, 307]:
            print("✅ Google OAuth redirect working")
        else:
            print(f"⚠️  OAuth redirect status: {response.status_code}")
    except:
        print("⚠️  OAuth endpoint error")
    
    # Test dashboard (should redirect to auth if not logged in)
    print("\n📊 Testing dashboard...")
    try:
        response = requests.get(f"{BASE_URL}/dashboard", allow_redirects=False)
        if response.status_code in [302, 307]:
            print("✅ Dashboard auth protection working")
        else:
            print(f"Dashboard status: {response.status_code}")
    except:
        print("⚠️  Dashboard endpoint error")
    
    # Test API endpoints (should require auth)
    print("\n📡 Testing API endpoints...")
    try:
        response = requests.get(f"{BASE_URL}/api/jobs/")
        if response.status_code == 401:
            print("✅ API auth protection working")
        else:
            print(f"API jobs status: {response.status_code}")
    except:
        print("⚠️  API endpoint error")
    
    print("\n" + "=" * 50)
    print("🌐 Service URLs:")
    print(f"🏠 Home: {BASE_URL}/")
    print(f"📊 Dashboard: {BASE_URL}/dashboard")
    print(f"📖 API Docs: {BASE_URL}/docs")
    print(f"🔧 Interactive API: {BASE_URL}/redoc")
    
    print("\n📝 Usage Instructions:")
    print("1. Visit the home page to sign in with Google")
    print("2. Go to dashboard to upload videos and manage jobs")
    print("3. Use API endpoints for programmatic access")
    
    print("\n⚙️  Setup Notes:")
    print("- Edit .env file with Google OAuth credentials")
    print("- Google Cloud Console: https://console.cloud.google.com/")
    print("- Set redirect URI: http://localhost:8000/auth/callback")

if __name__ == "__main__":
    test_async_service()
