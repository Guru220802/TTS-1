#!/usr/bin/env python3
"""
Test Emotional TTS System
Demonstrate the emotional TTS capabilities
"""

import os
import sys
import time
import requests
import json
from pathlib import Path


def test_api_endpoint(text: str, emotion: str = "balanced", user_persona: str = "youth"):
    """Test the TTS API endpoint"""
    url = "http://192.168.1.102:8002/api/generate-avatar"
    
    data = {
        "text": text,
        "target_lang": "en",
        "user_persona": user_persona
    }
    
    try:
        print(f"🎯 Testing emotion: {emotion}")
        print(f"   Text: {text}")
        print(f"   Persona: {user_persona}")
        
        response = requests.post(url, data=data, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Success: {result.get('video_url', 'No video URL')}")
            return True
        else:
            print(f"   ❌ Failed: {response.status_code} - {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Request failed: {e}")
        return False


def test_direct_tts():
    """Test TTS engines directly"""
    print("🧪 Testing TTS engines directly...")
    
    test_text = "Hello! This is a test of the emotional text-to-speech system."
    
    # Test LoRA TTS if available
    try:
        from lora_tts_engine import get_lora_tts_engine
        
        print("\n🎯 Testing LoRA TTS Engine:")
        lora_engine = get_lora_tts_engine()
        
        if lora_engine.is_available():
            for emotion in ["joyful", "peaceful", "enthusiastic"]:
                try:
                    output_path = f"test_lora_{emotion}.wav"
                    result = lora_engine.generate_speech(
                        text=test_text,
                        emotion=emotion,
                        output_path=output_path
                    )
                    print(f"   ✅ {emotion}: {result}")
                except Exception as e:
                    print(f"   ❌ {emotion}: {e}")
        else:
            print("   ⚠️ LoRA TTS not available")
            
    except ImportError:
        print("   ⚠️ LoRA TTS module not found")
    
    # Test Emotional Fallback TTS
    try:
        from emotional_fallback_tts import get_emotional_fallback_tts
        
        print("\n🔄 Testing Emotional Fallback TTS:")
        fallback_engine = get_emotional_fallback_tts()
        
        for emotion in ["confident", "soothing", "inspiring"]:
            try:
                result = fallback_engine.generate_emotional_speech(
                    text=test_text,
                    emotion=emotion
                )
                print(f"   ✅ {emotion}: {result}")
            except Exception as e:
                print(f"   ❌ {emotion}: {e}")
                
    except ImportError:
        print("   ⚠️ Emotional Fallback TTS module not found")


def test_all_emotions():
    """Test all available emotions through the API"""
    print("🎭 Testing all emotions through API...")
    
    emotions_texts = {
        "joyful": "What a wonderful day this is! I'm so excited to share this amazing news with you!",
        "peaceful": "Take a deep breath and let yourself relax completely in this quiet moment.",
        "balanced": "Today we'll explore some interesting concepts together in a structured way.",
        "enthusiastic": "This is absolutely incredible! You won't believe what we discovered!",
        "contemplative": "Let's pause for a moment to consider the deeper meaning of this experience.",
        "warm": "Welcome, dear friend. It's so good to have you here with us today.",
        "inspiring": "You have the power to change the world with your unique gifts and talents.",
        "confident": "I know exactly what needs to be done, and we will succeed together.",
        "grounded": "Let's focus on the practical steps we need to take moving forward.",
        "soothing": "Everything will be alright. You are safe and cared for in this moment."
    }
    
    success_count = 0
    total_count = len(emotions_texts)
    
    for emotion, text in emotions_texts.items():
        if test_api_endpoint(text, emotion):
            success_count += 1
        time.sleep(2)  # Brief pause between requests
    
    print(f"\n📊 Results: {success_count}/{total_count} emotions tested successfully")
    return success_count == total_count


def check_server_status():
    """Check if the TTS server is running"""
    try:
        response = requests.get("http://192.168.1.102:8002/", timeout=5)
        return response.status_code == 200
    except:
        return False


def main():
    """Main test function"""
    print("🎭 Emotional TTS Testing Suite")
    print("=" * 50)
    
    # Check if server is running
    if not check_server_status():
        print("❌ TTS server is not running!")
        print("   Please start the server first:")
        print("   python avatar_engine.py")
        sys.exit(1)
    
    print("✅ TTS server is running")
    
    # Test direct TTS engines
    test_direct_tts()
    
    # Test API with sample emotions
    print("\n🌐 Testing API endpoints...")
    
    sample_tests = [
        ("Hello! This is a joyful message!", "joyful", "youth"),
        ("Please relax and find your inner peace.", "peaceful", "spiritual"),
        ("Let's learn something new today!", "enthusiastic", "kids"),
        ("You can achieve anything you set your mind to.", "inspiring", "youth")
    ]
    
    for text, emotion, persona in sample_tests:
        test_api_endpoint(text, emotion, persona)
        time.sleep(1)
    
    # Test all emotions (optional)
    print("\n" + "=" * 50)
    response = input("🤔 Test all 10 emotions? This will take a few minutes. (y/N): ")
    
    if response.lower() in ['y', 'yes']:
        success = test_all_emotions()
        if success:
            print("🎉 All emotion tests passed!")
        else:
            print("⚠️ Some emotion tests failed. Check the logs for details.")
    
    print("\n✅ Testing complete!")
    print("\nGenerated files:")
    for file in Path(".").glob("test_*.wav"):
        print(f"   🎵 {file}")
    for file in Path(".").glob("test_*.mp3"):
        print(f"   🎵 {file}")


if __name__ == "__main__":
    main()
