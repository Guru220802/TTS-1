#!/usr/bin/env python3
"""
Emotional TTS Setup Script
Install dependencies and initialize the emotional TTS system
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


def run_command(command, description=""):
    """Run a command and handle errors"""
    print(f"🔧 {description}")
    print(f"   Running: {command}")
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        print(f"   ✅ Success")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Failed: {e}")
        print(f"   Error output: {e.stderr}")
        return False


def check_python_version():
    """Check Python version compatibility"""
    print("🐍 Checking Python version...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} is not supported")
        print("   Please use Python 3.8 or higher")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True


def check_system_dependencies():
    """Check system-level dependencies"""
    print("🔍 Checking system dependencies...")
    
    # Check for ffmpeg
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        print("✅ FFmpeg is available")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️ FFmpeg not found")
        print("   Please install FFmpeg:")
        if platform.system() == "Windows":
            print("   - Download from https://ffmpeg.org/download.html")
            print("   - Or use: winget install ffmpeg")
        elif platform.system() == "Darwin":  # macOS
            print("   - brew install ffmpeg")
        else:  # Linux
            print("   - sudo apt-get install ffmpeg  (Ubuntu/Debian)")
            print("   - sudo yum install ffmpeg      (CentOS/RHEL)")
    
    return True


def install_base_dependencies():
    """Install base Python dependencies"""
    print("📦 Installing base dependencies...")
    
    base_packages = [
        "fastapi>=0.104.1",
        "uvicorn>=0.24.0",
        "gtts>=2.3.0",
        "pydub>=0.25.1",
        "numpy>=1.24.0",
        "librosa>=0.10.1",
        "soundfile>=0.12.1",
        "scipy>=1.10.0"
    ]
    
    for package in base_packages:
        if not run_command(f"pip install {package}", f"Installing {package}"):
            return False
    
    return True


def install_lora_dependencies():
    """Install LoRA TTS dependencies"""
    print("🎯 Installing LoRA TTS dependencies...")
    
    # Install from requirements file if it exists
    if os.path.exists("requirements_lora_tts.txt"):
        return run_command(
            "pip install -r requirements_lora_tts.txt",
            "Installing LoRA TTS requirements"
        )
    
    # Fallback to individual packages
    lora_packages = [
        "TTS>=0.22.0",
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "peft>=0.7.0",
        "transformers>=4.36.0",
        "accelerate>=0.25.0"
    ]
    
    for package in lora_packages:
        if not run_command(f"pip install {package}", f"Installing {package}"):
            print(f"⚠️ Failed to install {package}, continuing...")
    
    return True


def create_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")
    
    directories = [
        "tts/tts_outputs",
        "lora_emotional_models",
        "emotional_training_data",
        "checkpoints",
        "logs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"   ✅ Created: {directory}")
    
    return True


def test_basic_functionality():
    """Test basic TTS functionality"""
    print("🧪 Testing basic functionality...")
    
    try:
        # Test gTTS
        from gtts import gTTS
        import tempfile
        
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tts = gTTS(text="Testing basic TTS functionality", lang='en')
            tts.save(tmp.name)
            
        os.unlink(tmp.name)
        print("   ✅ Basic gTTS working")
        
        # Test audio processing
        from pydub import AudioSegment
        print("   ✅ Audio processing available")
        
        # Test numpy and scipy
        import numpy as np
        import scipy.signal
        print("   ✅ Scientific computing libraries available")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Basic functionality test failed: {e}")
        return False


def test_lora_functionality():
    """Test LoRA TTS functionality"""
    print("🎯 Testing LoRA functionality...")
    
    try:
        # Test TTS library
        from TTS.api import TTS
        print("   ✅ Coqui TTS available")
        
        # Test PEFT
        from peft import LoraConfig
        print("   ✅ PEFT (LoRA) available")
        
        # Test torch
        import torch
        print(f"   ✅ PyTorch available (CUDA: {torch.cuda.is_available()})")
        
        return True
        
    except ImportError as e:
        print(f"   ⚠️ LoRA functionality not fully available: {e}")
        print("   LoRA TTS will fall back to enhanced gTTS")
        return False


def create_sample_config():
    """Create sample configuration files"""
    print("⚙️ Creating sample configuration...")
    
    # Training configuration
    training_config = {
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "learning_rate": 1e-4,
        "batch_size": 2,
        "num_epochs": 5,
        "emotions": [
            "joyful", "peaceful", "balanced", "enthusiastic", "contemplative",
            "warm", "inspiring", "confident", "grounded", "soothing"
        ],
        "sample_rate": 22050,
        "data_dir": "emotional_training_data",
        "output_dir": "lora_emotional_models"
    }
    
    with open("emotional_training_config.json", "w") as f:
        import json
        json.dump(training_config, f, indent=2)
    
    print("   ✅ Created: emotional_training_config.json")
    
    # Create README
    readme_content = """# Emotional TTS System

## Quick Start

1. **Test the system:**
   ```bash
   python avatar_engine.py
   ```

2. **Train emotional LoRA (optional):**
   ```bash
   # Create training data
   python train_emotional_lora.py create-data
   
   # Train the model
   python train_emotional_lora.py train
   
   # Test the trained model
   python train_emotional_lora.py test --emotion joyful
   ```

3. **Use the API:**
   - Start the server: `python avatar_engine.py`
   - Open browser: `http://192.168.1.102:8002`
   - Send POST request to `/api/generate-avatar`

## Features

- ✅ **LoRA TTS**: Advanced neural TTS with emotional control
- ✅ **Emotional Fallback**: Enhanced gTTS with emotional processing
- ✅ **10 Emotions**: joyful, peaceful, balanced, enthusiastic, contemplative, warm, inspiring, confident, grounded, soothing
- ✅ **Transition Tones**: Emotion-specific audio tones
- ✅ **Audio Optimization**: Compression and enhancement
- ✅ **Sentiment Analysis**: Automatic emotion detection

## Troubleshooting

- If LoRA TTS fails, the system automatically falls back to enhanced gTTS
- Check logs in `emotional_lora_training.log` for training issues
- Ensure FFmpeg is installed for audio processing
"""
    
    with open("EMOTIONAL_TTS_README.md", "w") as f:
        f.write(readme_content)
    
    print("   ✅ Created: EMOTIONAL_TTS_README.md")
    
    return True


def main():
    """Main setup function"""
    print("🎭 Emotional TTS Setup")
    print("=" * 50)
    
    # Check prerequisites
    if not check_python_version():
        sys.exit(1)
    
    check_system_dependencies()
    
    # Install dependencies
    print("\n📦 Installing Dependencies")
    print("-" * 30)
    
    if not install_base_dependencies():
        print("❌ Failed to install base dependencies")
        sys.exit(1)
    
    install_lora_dependencies()  # Continue even if this fails
    
    # Setup environment
    print("\n🔧 Setting Up Environment")
    print("-" * 30)
    
    create_directories()
    create_sample_config()
    
    # Test functionality
    print("\n🧪 Testing Functionality")
    print("-" * 30)
    
    basic_ok = test_basic_functionality()
    lora_ok = test_lora_functionality()
    
    # Summary
    print("\n📋 Setup Summary")
    print("-" * 30)
    
    print(f"✅ Basic TTS: {'Working' if basic_ok else 'Failed'}")
    print(f"{'✅' if lora_ok else '⚠️'} LoRA TTS: {'Available' if lora_ok else 'Fallback mode'}")
    
    if basic_ok:
        print("\n🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Run: python avatar_engine.py")
        print("2. Open: http://192.168.1.102:8002")
        print("3. (Optional) Train LoRA: python train_emotional_lora.py create-data")
        print("\nSee EMOTIONAL_TTS_README.md for detailed instructions.")
    else:
        print("\n❌ Setup failed. Please check the error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
