#!/usr/bin/env python3
"""
Test script to verify the original audio UI feature is working
"""
import asyncio
import aiohttp
import json
from datetime import datetime

async def test_original_audio_ui():
    """Test the original audio feature via the API"""
    
    # Test URL - using a short YouTube video
    test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Rick Roll (short video)
    
    async with aiohttp.ClientSession() as session:
        # Test 1: Process video URL with original audio enabled
        print("🧪 Testing Video URL processing with original audio...")
        
        payload = {
            "url": test_url,
            "summary_prompt": "Test summary for UI verification",
            "keep_original_audio": True
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer test@example.com"  # Test user
        }
        
        try:
            async with session.post(
                "http://localhost:8000/api/process-video-url/",
                json=payload,
                headers=headers
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    print(f"✅ Video URL processing started successfully")
                    print(f"   Job ID: {result.get('job_id')}")
                    print(f"   Keep Original Audio: {payload['keep_original_audio']}")
                    return result.get('job_id')
                else:
                    error_text = await response.text()
                    print(f"❌ Video URL processing failed: {response.status}")
                    print(f"   Error: {error_text}")
                    return None
                    
        except Exception as e:
            print(f"❌ Request failed: {e}")
            return None

async def check_job_status(job_id):
    """Check the job status and verify original audio field"""
    if not job_id:
        return
        
    async with aiohttp.ClientSession() as session:
        headers = {
            "Authorization": "Bearer test@example.com"
        }
        
        try:
            async with session.get(
                f"http://localhost:8000/api/jobs/{job_id}",
                headers=headers
            ) as response:
                if response.status == 200:
                    job_data = await response.json()
                    print(f"📋 Job Status: {job_data.get('status')}")
                    print(f"   Has Original Audio File: {bool(job_data.get('original_audio_file'))}")
                    if job_data.get('original_audio_file'):
                        print(f"   Original Audio File: {job_data.get('original_audio_file')}")
                        print(f"✅ Original audio feature is working in the API!")
                    return job_data
                else:
                    error_text = await response.text()
                    print(f"❌ Job status check failed: {response.status}")
                    print(f"   Error: {error_text}")
                    
        except Exception as e:
            print(f"❌ Job status check failed: {e}")

async def main():
    """Main test function"""
    print("🚀 Testing Original Audio UI Feature")
    print("=" * 50)
    
    # Test the API endpoint
    job_id = await test_original_audio_ui()
    
    if job_id:
        print("\n⏳ Waiting a moment for job to process...")
        await asyncio.sleep(2)
        
        # Check job status
        await check_job_status(job_id)
        
        print("\n📋 UI Elements to verify manually:")
        print("   1. Video URL form should have 'Keep Original Audio' checkbox")
        print("   2. Video Upload form should have 'Keep Original Audio' checkbox")
        print("   3. Completed jobs should show 'Play Original Audio' option if available")
        print("   4. Download original audio should be available in job actions")
        
    print("\n🌐 Open http://localhost:8000 to test the UI manually")
    print("✅ UI test setup complete!")

if __name__ == "__main__":
    asyncio.run(main())
