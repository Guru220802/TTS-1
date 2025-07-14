#!/usr/bin/env python3
"""
Emotional LoRA Training Script
Train LoRA models for emotional TTS control
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import json

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from lora_emotional_trainer import (
    EmotionalLoRATrainer, 
    EmotionalTrainingConfig,
    create_emotional_training_samples
)


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('emotional_lora_training.log')
        ]
    )


def create_sample_training_data(output_dir: str):
    """Create sample training data for emotional TTS"""
    print("🎯 Creating sample training data...")
    
    # Get sample texts and emotions
    texts, emotions = create_emotional_training_samples()
    
    print(f"Generated {len(texts)} training samples across {len(set(emotions))} emotions")
    
    # Create training configuration
    config = EmotionalTrainingConfig(
        data_dir=output_dir,
        output_dir="lora_emotional_models"
    )
    
    # Initialize trainer
    trainer = EmotionalLoRATrainer(config)
    
    try:
        # Setup models
        print("🚀 Setting up TTS models...")
        trainer.setup_models()
        
        # Generate training data
        print("🎵 Generating audio training data...")
        trainer.create_training_data(texts, emotions, output_dir)
        
        print(f"✅ Training data created in: {output_dir}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to create training data: {e}")
        return False


def train_emotional_lora(data_dir: str, output_dir: str, config_file: str = None):
    """Train emotional LoRA model"""
    print("🎯 Starting emotional LoRA training...")
    
    # Load configuration
    if config_file and os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config_dict = json.load(f)
        config = EmotionalTrainingConfig(**config_dict)
    else:
        config = EmotionalTrainingConfig(
            data_dir=data_dir,
            output_dir=output_dir,
            lora_r=16,
            lora_alpha=32,
            learning_rate=1e-4,
            batch_size=2,
            num_epochs=5
        )
    
    # Initialize trainer
    trainer = EmotionalLoRATrainer(config)
    
    try:
        # Setup models
        print("🚀 Setting up models for training...")
        trainer.setup_models()
        
        # Check if training data exists
        if not os.path.exists(data_dir):
            print(f"⚠️ Training data directory not found: {data_dir}")
            print("Creating sample training data...")
            if not create_sample_training_data(data_dir):
                return False
        
        # Train LoRA
        print("🎓 Starting LoRA training...")
        final_loss = trainer.train_emotional_lora(data_dir)
        
        # Save trained model
        print("💾 Saving trained LoRA model...")
        trainer.save_lora_weights(output_dir)
        
        print(f"✅ Training completed successfully!")
        print(f"   Final loss: {final_loss:.4f}")
        print(f"   Model saved to: {output_dir}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_emotional_lora(model_dir: str, test_text: str = None, emotion: str = "joyful"):
    """Test trained emotional LoRA model"""
    print("🧪 Testing emotional LoRA model...")
    
    if test_text is None:
        test_text = "Hello! This is a test of the emotional text-to-speech system."
    
    try:
        # Import TTS engine
        from lora_tts_engine import LoRATTSEngine
        
        # Initialize engine
        engine = LoRATTSEngine()
        
        # Load trained weights
        if os.path.exists(model_dir):
            success = engine.load_lora_weights(model_dir)
            if not success:
                print(f"❌ Failed to load LoRA weights from {model_dir}")
                return False
        else:
            print(f"⚠️ Model directory not found: {model_dir}")
            return False
        
        # Generate test audio
        output_path = f"test_emotional_output_{emotion}.wav"
        result_path = engine.generate_speech(
            text=test_text,
            emotion=emotion,
            output_path=output_path
        )
        
        print(f"✅ Test audio generated: {result_path}")
        print(f"   Text: {test_text}")
        print(f"   Emotion: {emotion}")
        
        return True
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main training script"""
    parser = argparse.ArgumentParser(description="Train emotional LoRA TTS models")
    
    parser.add_argument(
        "command",
        choices=["create-data", "train", "test"],
        help="Command to execute"
    )
    
    parser.add_argument(
        "--data-dir",
        default="emotional_training_data",
        help="Directory for training data"
    )
    
    parser.add_argument(
        "--output-dir",
        default="lora_emotional_models",
        help="Directory to save trained model"
    )
    
    parser.add_argument(
        "--config",
        help="Training configuration file (JSON)"
    )
    
    parser.add_argument(
        "--test-text",
        help="Text to use for testing"
    )
    
    parser.add_argument(
        "--emotion",
        default="joyful",
        choices=[
            "joyful", "peaceful", "balanced", "enthusiastic", "contemplative",
            "warm", "inspiring", "confident", "grounded", "soothing"
        ],
        help="Emotion for testing"
    )
    
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    print("🎭 Emotional LoRA TTS Training System")
    print("=" * 50)
    
    # Execute command
    if args.command == "create-data":
        success = create_sample_training_data(args.data_dir)
        
    elif args.command == "train":
        success = train_emotional_lora(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            config_file=args.config
        )
        
    elif args.command == "test":
        success = test_emotional_lora(
            model_dir=args.output_dir,
            test_text=args.test_text,
            emotion=args.emotion
        )
    
    else:
        print(f"❌ Unknown command: {args.command}")
        success = False
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
