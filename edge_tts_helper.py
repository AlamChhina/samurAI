#!/usr/bin/env python3
"""
Simple Edge-TTS Voice Configuration Helper
Shows available Microsoft Neural Voices for edge-tts
"""

import asyncio
import edge_tts
import os
from dotenv import load_dotenv

load_dotenv()

async def list_edge_tts_voices():
    """List all available edge-tts neural voices"""
    print("🎤 Available Edge-TTS Neural Voices (Microsoft):")
    print("=" * 60)
    
    try:
        voices = await edge_tts.list_voices()
        
        # Filter for English neural voices only
        english_neural_voices = [
            voice for voice in voices 
            if voice["Locale"].startswith("en-") and "Neural" in voice["VoiceTag"]
        ]
        
        # Sort by locale and gender
        english_neural_voices.sort(key=lambda x: (x["Locale"], x["Gender"]))
        
        current_locale = None
        for voice in english_neural_voices:
            if voice["Locale"] != current_locale:
                current_locale = voice["Locale"]
                print(f"\n📍 {current_locale}:")
                
            gender_emoji = "👨" if voice["Gender"] == "Male" else "👩"
            print(f"  {gender_emoji} {voice['ShortName']:<35} - {voice['FriendlyName']}")
            
            # Highlight special voices
            if "Multilingual" in voice["VoiceTag"]:
                print("      ⭐ PREMIUM: Multilingual support")
            if any(tag in voice["VoiceTag"] for tag in ["Conversational", "Chat"]):
                print("      ✨ SPECIALTY: Conversational")
                
    except Exception as e:
        print(f"Error listing voices: {e}")

def show_current_config():
    """Show current configuration"""
    print("\n⚙️ Current Configuration:")
    print("=" * 60)
    
    preferred_voice = os.getenv("PREFERRED_VOICE", "en-US-AriaNeural")
    print(f"Current Voice: {preferred_voice}")
    
    print("\n🎯 Recommended Voices by Use Case:")
    print("   📚 Audiobooks/Long content: en-US-JennyMultilingualNeural")
    print("   🎥 Video narration: en-US-AriaNeural")
    print("   📢 Announcements: en-US-DavisNeural (male)")
    print("   💼 Professional: en-US-MonicaNeural")
    print("   🎧 Podcasts: en-US-SaraNeural")
    
    print("\n🔧 To change voice:")
    print("   Edit .env file: PREFERRED_VOICE=your-chosen-voice")

async def test_current_voice():
    """Test current voice configuration"""
    print("\n🧪 Testing Current Voice Configuration:")
    print("=" * 60)
    
    test_text = "Hello! This is a test of the Edge-TTS neural voice. The quality should be clear and natural."
    output_file = "edge_tts_test.mp3"
    
    try:
        import sys
        sys.path.append('.')
        from async_video_service import generate_edge_tts
        
        preferred_voice = os.getenv("PREFERRED_VOICE", "en-US-AriaNeural")
        success = await generate_edge_tts(test_text, output_file, preferred_voice)
        
        if success and os.path.exists(output_file):
            file_size = os.path.getsize(output_file)
            print(f"✅ SUCCESS: Generated {output_file} ({file_size:,} bytes)")
            print(f"🎤 Voice used: {preferred_voice}")
            
            # Auto-play on macOS
            import platform
            if platform.system() == "Darwin":
                import subprocess
                try:
                    subprocess.run(["afplay", output_file], check=False)
                    print("🔊 Playing audio sample...")
                except Exception:
                    print("   (You can manually open the file to listen)")
        else:
            print("❌ FAILED: Could not generate test audio")
            
    except Exception as e:
        print(f"❌ Error testing voice: {e}")

async def main():
    print("🎵 Edge-TTS Voice Configuration Helper")
    print("Microsoft Neural Voices - High Quality & Free!\n")
    
    # Show available voices
    await list_edge_tts_voices()
    show_current_config()
    
    # Ask if user wants to test
    print("\n" + "=" * 60)
    try:
        response = input("Test current voice configuration? (y/n): ").strip().lower()
        if response == 'y':
            await test_current_voice()
    except KeyboardInterrupt:
        print("\n\nConfiguration helper completed!")

if __name__ == "__main__":
    asyncio.run(main())
