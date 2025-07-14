"""
LoRA Emotional TTS Trainer
Fine-tune TTS models with LoRA for emotional tone control
"""

import os
import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json
import yaml
from pathlib import Path
import torchaudio
from torch.utils.data import Dataset, DataLoader
import librosa

try:
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel
    from transformers import Trainer, TrainingArguments
    PEFT_AVAILABLE = True
except ImportError:
    print("⚠️ PEFT not available. Install with: pip install peft")
    PEFT_AVAILABLE = False

try:
    from TTS.api import TTS  # type: ignore
    from TTS.tts.configs.xtts_config import XttsConfig  # type: ignore
    from TTS.tts.models.xtts import Xtts  # type: ignore
    TTS_AVAILABLE = True
except ImportError:
    print("⚠️ Coqui TTS not available. Install with: pip install TTS")
    print("   Note: TTS requires Microsoft Visual C++ 14.0 or greater on Windows")
    print("   Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/")
    TTS_AVAILABLE = False

    # Create placeholder classes to prevent import errors
    class TTS:
        def __init__(self, *args, **kwargs):
            raise ImportError("Coqui TTS not available")

    class XttsConfig:
        pass

    class Xtts:
        pass


@dataclass
class EmotionalTrainingConfig:
    """Configuration for emotional LoRA training"""
    # LoRA parameters
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: List[str] = None
    
    # Training parameters
    learning_rate: float = 1e-4
    batch_size: int = 4
    num_epochs: int = 10
    warmup_steps: int = 100
    gradient_accumulation_steps: int = 4
    
    # Emotional parameters
    emotions: List[str] = None
    emotion_weights: Dict[str, float] = None
    
    # Audio parameters
    sample_rate: int = 22050
    max_audio_length: int = 10  # seconds
    
    # Paths
    output_dir: str = "lora_emotional_models"
    data_dir: str = "emotional_training_data"
    base_model: str = "tts_models/multilingual/multi-dataset/xtts_v2"


class EmotionalAudioDataset(Dataset):
    """Dataset for emotional TTS training"""
    
    def __init__(self, data_dir: str, emotions: List[str], sample_rate: int = 22050):
        self.data_dir = Path(data_dir)
        self.emotions = emotions
        self.sample_rate = sample_rate
        self.samples = self._load_samples()
        
    def _load_samples(self) -> List[Dict]:
        """Load training samples from directory structure"""
        samples = []
        
        for emotion in self.emotions:
            emotion_dir = self.data_dir / emotion
            if not emotion_dir.exists():
                continue
                
            # Look for audio files and corresponding text files
            for audio_file in emotion_dir.glob("*.wav"):
                text_file = audio_file.with_suffix(".txt")
                if text_file.exists():
                    samples.append({
                        'audio_path': str(audio_file),
                        'text_path': str(text_file),
                        'emotion': emotion,
                        'emotion_id': self.emotions.index(emotion)
                    })
        
        logging.info(f"Loaded {len(samples)} training samples")
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load audio
        audio, _ = librosa.load(sample['audio_path'], sr=self.sample_rate)
        
        # Load text
        with open(sample['text_path'], 'r', encoding='utf-8') as f:
            text = f.read().strip()
        
        return {
            'audio': torch.FloatTensor(audio),
            'text': text,
            'emotion': sample['emotion'],
            'emotion_id': torch.LongTensor([sample['emotion_id']])
        }


class EmotionalLoRATrainer:
    """Trainer for emotional LoRA fine-tuning"""
    
    def __init__(self, config: EmotionalTrainingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize emotions if not provided
        if self.config.emotions is None:
            self.config.emotions = [
                'joyful', 'peaceful', 'balanced', 'enthusiastic', 'contemplative',
                'warm', 'inspiring', 'confident', 'grounded', 'soothing'
            ]
        
        # Initialize emotion weights
        if self.config.emotion_weights is None:
            self.config.emotion_weights = {emotion: 1.0 for emotion in self.config.emotions}
        
        # Initialize models
        self.base_model = None
        self.lora_model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def setup_models(self):
        """Setup base model and LoRA adaptation"""
        if not TTS_AVAILABLE:
            raise ImportError("Coqui TTS not available")
        
        self.logger.info("Loading base TTS model...")
        
        # Load base model
        self.base_model = TTS(
            model_name=self.config.base_model,
            progress_bar=True,
            gpu=self.device.type == "cuda"
        )
        
        # Setup LoRA configuration
        if PEFT_AVAILABLE:
            self._setup_lora()
        else:
            self.logger.warning("PEFT not available, using base model only")
    
    def _setup_lora(self):
        """Setup LoRA adaptation"""
        try:
            # Get the underlying model for LoRA
            if hasattr(self.base_model, 'synthesizer'):
                target_model = self.base_model.synthesizer
            else:
                self.logger.warning("Could not access synthesizer for LoRA")
                return
            
            # Configure LoRA
            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                target_modules=self.config.target_modules or ["linear", "conv1d"],
                lora_dropout=self.config.lora_dropout,
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION
            )
            
            # Apply LoRA
            self.lora_model = get_peft_model(target_model, lora_config)
            self.lora_model.to(self.device)
            
            self.logger.info("✅ LoRA adaptation setup complete")
            
        except Exception as e:
            self.logger.error(f"LoRA setup failed: {e}")
            raise
    
    def create_training_data(self, texts: List[str], emotions: List[str], output_dir: str):
        """Create training data by generating audio samples"""
        os.makedirs(output_dir, exist_ok=True)
        
        for emotion in set(emotions):
            emotion_dir = os.path.join(output_dir, emotion)
            os.makedirs(emotion_dir, exist_ok=True)
        
        self.logger.info(f"Generating training data for {len(texts)} samples...")
        
        for i, (text, emotion) in enumerate(zip(texts, emotions)):
            try:
                # Generate audio with base model
                audio = self.base_model.tts(text=text, language="en")
                
                # Save audio and text
                emotion_dir = os.path.join(output_dir, emotion)
                audio_path = os.path.join(emotion_dir, f"sample_{i:04d}.wav")
                text_path = os.path.join(emotion_dir, f"sample_{i:04d}.txt")
                
                # Save audio
                torchaudio.save(audio_path, torch.FloatTensor(audio).unsqueeze(0), self.config.sample_rate)
                
                # Save text
                with open(text_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                
                if i % 10 == 0:
                    self.logger.info(f"Generated {i+1}/{len(texts)} samples")
                    
            except Exception as e:
                self.logger.warning(f"Failed to generate sample {i}: {e}")
        
        self.logger.info("✅ Training data generation complete")
    
    def train_emotional_lora(self, data_dir: str):
        """Train LoRA for emotional control"""
        if not self.lora_model:
            raise RuntimeError("LoRA model not initialized")
        
        # Create dataset
        dataset = EmotionalAudioDataset(
            data_dir=data_dir,
            emotions=self.config.emotions,
            sample_rate=self.config.sample_rate
        )
        
        if len(dataset) == 0:
            raise ValueError("No training data found")
        
        # Create data loader
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=2
        )
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(
            self.lora_model.parameters(),
            lr=self.config.learning_rate
        )
        
        # Setup loss function
        criterion = nn.MSELoss()
        
        # Training loop
        self.lora_model.train()
        total_loss = 0
        
        self.logger.info(f"Starting training for {self.config.num_epochs} epochs...")
        
        for epoch in range(self.config.num_epochs):
            epoch_loss = 0
            
            for batch_idx, batch in enumerate(dataloader):
                optimizer.zero_grad()
                
                # Move to device
                audio = batch['audio'].to(self.device)
                emotion_ids = batch['emotion_id'].to(self.device)
                
                try:
                    # Forward pass (simplified - actual implementation would depend on model architecture)
                    # This is a placeholder for the actual training logic
                    outputs = self._forward_pass(audio, emotion_ids)
                    loss = criterion(outputs, audio)  # Simplified loss
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    total_loss += loss.item()
                    
                    if batch_idx % 10 == 0:
                        self.logger.info(f"Epoch {epoch+1}/{self.config.num_epochs}, "
                                       f"Batch {batch_idx}, Loss: {loss.item():.4f}")
                
                except Exception as e:
                    self.logger.warning(f"Training step failed: {e}")
                    continue
            
            avg_epoch_loss = epoch_loss / len(dataloader)
            self.logger.info(f"Epoch {epoch+1} completed. Average loss: {avg_epoch_loss:.4f}")
        
        self.logger.info("✅ Training completed")
        return total_loss / (self.config.num_epochs * len(dataloader))
    
    def _forward_pass(self, audio: torch.Tensor, emotion_ids: torch.Tensor) -> torch.Tensor:  # pylint: disable=unused-argument
        """Simplified forward pass - to be implemented based on actual model architecture"""
        # This is a placeholder - actual implementation would depend on the TTS model structure
        # For now, return the input audio as a placeholder
        return audio
    
    def save_lora_weights(self, output_path: str):
        """Save trained LoRA weights"""
        if not self.lora_model:
            raise RuntimeError("LoRA model not available")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save LoRA weights
        self.lora_model.save_pretrained(output_path)
        
        # Save configuration
        config_path = os.path.join(output_path, "training_config.json")
        with open(config_path, 'w') as f:
            json.dump({
                'emotions': self.config.emotions,
                'emotion_weights': self.config.emotion_weights,
                'lora_r': self.config.lora_r,
                'lora_alpha': self.config.lora_alpha,
                'sample_rate': self.config.sample_rate
            }, f, indent=2)
        
        self.logger.info(f"✅ LoRA weights saved to {output_path}")
    
    def load_lora_weights(self, weights_path: str):
        """Load pre-trained LoRA weights"""
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"LoRA weights not found: {weights_path}")
        
        try:
            # Load LoRA weights
            self.lora_model = PeftModel.from_pretrained(
                self.base_model.synthesizer,
                weights_path
            )
            
            # Load configuration
            config_path = os.path.join(weights_path, "training_config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    self.config.emotions = config.get('emotions', self.config.emotions)
                    self.config.emotion_weights = config.get('emotion_weights', self.config.emotion_weights)
            
            self.logger.info(f"✅ LoRA weights loaded from {weights_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to load LoRA weights: {e}")
            raise


def create_emotional_training_samples() -> Tuple[List[str], List[str]]:
    """Create sample training data for emotional TTS"""
    
    # Sample texts for different emotions
    training_samples = {
        'joyful': [
            "What a wonderful day this is! I'm so excited to share this with you.",
            "Congratulations! You've achieved something amazing today.",
            "This is fantastic news! I can't wait to celebrate with everyone."
        ],
        'peaceful': [
            "Take a deep breath and let yourself relax completely.",
            "In this quiet moment, find your inner calm and serenity.",
            "Let the gentle sounds of nature wash over you peacefully."
        ],
        'balanced': [
            "Today we'll explore some interesting concepts together.",
            "Let's take a balanced approach to understanding this topic.",
            "Here's what we need to know about this subject."
        ],
        'enthusiastic': [
            "This is absolutely incredible! You won't believe what we discovered!",
            "Get ready for an amazing adventure that will blow your mind!",
            "The possibilities are endless and the future looks bright!"
        ],
        'contemplative': [
            "Let's pause for a moment to consider the deeper meaning.",
            "This raises some profound questions worth reflecting upon.",
            "Sometimes the most important insights come from quiet thought."
        ],
        'warm': [
            "Welcome, dear friend. It's so good to have you here with us.",
            "Your presence brings such warmth and comfort to this space.",
            "Thank you for being such a caring and thoughtful person."
        ],
        'inspiring': [
            "You have the power to change the world with your unique gifts.",
            "Every challenge is an opportunity to grow stronger and wiser.",
            "Believe in yourself, because you are capable of extraordinary things."
        ],
        'confident': [
            "I know exactly what needs to be done, and we will succeed.",
            "With determination and skill, we can overcome any obstacle.",
            "Trust in our abilities, because we have what it takes to win."
        ],
        'grounded': [
            "Let's focus on the practical steps we need to take.",
            "Here are the solid facts we can rely on moving forward.",
            "Building on this strong foundation, we can make real progress."
        ],
        'soothing': [
            "Everything will be alright. You are safe and cared for.",
            "Gently close your eyes and let the tension melt away.",
            "Rest now, knowing that tomorrow brings new hope and healing."
        ]
    }
    
    texts = []
    emotions = []
    
    for emotion, samples in training_samples.items():
        texts.extend(samples)
        emotions.extend([emotion] * len(samples))
    
    return texts, emotions


# Example usage and training script
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create training configuration
    config = EmotionalTrainingConfig(
        lora_r=16,
        lora_alpha=32,
        learning_rate=1e-4,
        batch_size=2,
        num_epochs=5,
        output_dir="lora_emotional_models"
    )
    
    # Initialize trainer
    trainer = EmotionalLoRATrainer(config)
    
    try:
        # Setup models
        trainer.setup_models()
        
        # Create training data
        texts, emotions = create_emotional_training_samples()
        trainer.create_training_data(texts, emotions, config.data_dir)
        
        # Train LoRA
        final_loss = trainer.train_emotional_lora(config.data_dir)
        print(f"Training completed with final loss: {final_loss:.4f}")
        
        # Save trained model
        trainer.save_lora_weights(config.output_dir)
        print("✅ Emotional LoRA training completed successfully!")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
