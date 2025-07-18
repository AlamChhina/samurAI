#!/usr/bin/env python3
"""
Test script for the original audio feature
"""
import asyncio
import os
from async_video_service import download_original_audio_from_url

async def test_original_audio():
    """Test downloading original audio from a YouTube video"""
    # Use a short public domain video for testing
    test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Rick Astley - Never Gonna Give You Up
    
    print(f"Testing original audio download from: {test_url}")
    
    result = await download_original_audio_from_url(test_url)
    
    if result:
        audio_path, title = result
        print(f"✅ Successfully downloaded: {title}")
        print(f"📁 File path: {audio_path}")
        
        if os.path.exists(audio_path):
            file_size = os.path.getsize(audio_path)
            print(f"📊 File size: {file_size} bytes")
            
            # Clean up the test file
            os.remove(audio_path)
            print("🧹 Cleaned up test file")
        else:
            print("❌ File not found at expected path")
    else:
        print("❌ Failed to download original audio")

if __name__ == "__main__":
    asyncio.run(test_original_audio())
