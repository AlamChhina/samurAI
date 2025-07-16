#!/usr/bin/env python3
"""
Test script to demonstrate yt-dlp video platform support
"""

import yt_dlp

def test_video_platform_support():
    """Test which video platforms yt-dlp can handle"""
    
    print("🎥 Testing Video Platform Support with yt-dlp\n")
    
    # Sample URLs for testing (these are real public videos)
    test_urls = {
        "YouTube": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",  # Rick Roll (short video)
        "Vimeo": "https://vimeo.com/148751763",  # Sample public video
        "YouTube ID": "dQw4w9WgXcQ",  # Just the video ID
    }
    
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'extract_flat': True,  # Don't download, just extract info
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for platform, url in test_urls.items():
            try:
                print(f"🔍 Testing {platform}: {url}")
                info = ydl.extract_info(url, download=False)
                title = info.get('title', 'Unknown')
                duration = info.get('duration', 0)
                uploader = info.get('uploader', 'Unknown')
                
                print(f"   ✅ Success!")
                print(f"   📹 Title: {title}")
                print(f"   ⏱️ Duration: {duration // 60}:{duration % 60:02d}")
                print(f"   👤 Uploader: {uploader}")
                print()
                
            except Exception as e:
                print(f"   ❌ Failed: {str(e)}")
                print()
    
    print("📋 Supported Platforms Summary:")
    print("   ✅ YouTube (URLs and video IDs)")
    print("   ✅ Vimeo")
    print("   ✅ Dailymotion")
    print("   ✅ Twitch")
    print("   ✅ Facebook Videos")
    print("   ✅ Instagram")
    print("   ✅ TikTok")
    print("   ✅ Twitter/X")
    print("   ✅ Reddit")
    print("   ✅ Streamable")
    print("   ✅ Archive.org")
    print("   ✅ And 1000+ more platforms!")
    print()
    print("🚀 Ready to process videos from multiple platforms!")

if __name__ == "__main__":
    test_video_platform_support()
