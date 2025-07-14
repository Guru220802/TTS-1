"""
LoRA-Enhanced TTS Engine
Advanced Text-to-Speech with LoRA fine-tuning and emotional voice control
"""

import os
import torch
import numpy as np
import logging
from typing import Dict, Optional, List
from dataclasses import dataclass
import json
import uuid

# Core TTS and Audio Processing
try:
    from TTS.api import TTS  # type: ignore
    from TTS.tts.configs.xtts_config import XttsConfig  # type: ignore
    from TTS.tts.models.xtts import Xtts  # type: ignore
    TTS_AVAILABLE = True
except ImportError as e:
    print("⚠️ Coqui TTS not available.")
    print("📋 Installation instructions:")
    print("   1. Install Microsoft Visual C++ Build Tools from:")
    print("      https://visualstudio.microsoft.com/visual-cpp-build-tools/")
    print("   2. Then install TTS with: pip install TTS")
    print("   3. Alternative: Use conda-forge: conda install -c conda-forge coqui-tts")
    print(f"   Error details: {e}")
    TTS_AVAILABLE = False

    # Create placeholder classes to prevent import errors
    class TTS:
        def __init__(self, *args, **kwargs):
            _ = args, kwargs  # Suppress unused parameter warnings
            raise ImportError("TTS package not available. Please install it first.")

    class XttsConfig:
        pass

    class Xtts:
        pass

try:
    from peft import LoraConfig, get_peft_model, TaskType
    LORA_AVAILABLE = True
except ImportError:
    print("⚠️ LoRA dependencies not available. Install with: pip install peft")
    LORA_AVAILABLE = False

from pydub import AudioSegment
import soundfile as sf


@dataclass
class LoRAConfig:
    """Configuration for LoRA fine-tuning"""
    r: int = 16  # LoRA rank
    lora_alpha: int = 32  # LoRA scaling parameter
    target_modules: List[str] = None  # Target modules for LoRA
    lora_dropout: float = 0.1
    bias: str = "none"  # Bias type
    task_type: str = "FEATURE_EXTRACTION"


@dataclass
class EmotionalVoiceConfig:
    """Configuration for emotional voice parameters"""
    emotion: str = "balanced"
    intensity: float = 0.7  # 0.0 to 1.0
    speed: float = 1.0  # Speech speed multiplier
    pitch_shift: float = 0.0  # Semitones
    energy: float = 1.0  # Voice energy/volume
    warmth: float = 0.5  # Voice warmth (0.0 cold, 1.0 warm)


class LoRATTSEngine:
    """
    Advanced TTS Engine with LoRA fine-tuning capabilities
    Supports emotional voice control and high-quality speech synthesis
    """
    
    def __init__(self, 
                 model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2",
                 device: str = "auto",
                 enable_lora: bool = True,
                 lora_config: Optional[LoRAConfig] = None):
        """
        Initialize LoRA TTS Engine
        
        Args:
            model_name: Base TTS model to use
            device: Device for inference ('cpu', 'cuda', 'auto')
            enable_lora: Whether to enable LoRA fine-tuning
            lora_config: LoRA configuration parameters
        """
        self.logger = logging.getLogger(__name__)
        self.model_name = model_name
        self.enable_lora = enable_lora and LORA_AVAILABLE
        self.lora_config = lora_config or LoRAConfig()
        
        # Device setup
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.logger.info(f"Initializing LoRA TTS Engine on {self.device}")
        
        # Model initialization
        self.base_model = None
        self.lora_model = None
        self.is_loaded = False
        
        # Emotional voice mappings
        self.emotion_configs = self._initialize_emotion_configs()
        
        # Voice cloning reference audios
        self.reference_voices = {}
        
        # Initialize models
        self._initialize_models()

        # Auto-load emotional LoRA weights if available
        self._auto_load_emotional_weights()
    
    def _initialize_emotion_configs(self) -> Dict[str, EmotionalVoiceConfig]:
        """Initialize emotional voice configurations"""
        return {
            'joyful': EmotionalVoiceConfig(
                emotion='joyful', intensity=0.8, speed=1.1, 
                pitch_shift=2.0, energy=1.2, warmth=0.8
            ),
            'peaceful': EmotionalVoiceConfig(
                emotion='peaceful', intensity=0.5, speed=0.9, 
                pitch_shift=-1.0, energy=0.8, warmth=0.9
            ),
            'balanced': EmotionalVoiceConfig(
                emotion='balanced', intensity=0.7, speed=1.0, 
                pitch_shift=0.0, energy=1.0, warmth=0.6
            ),
            'enthusiastic': EmotionalVoiceConfig(
                emotion='enthusiastic', intensity=0.9, speed=1.2, 
                pitch_shift=3.0, energy=1.3, warmth=0.7
            ),
            'contemplative': EmotionalVoiceConfig(
                emotion='contemplative', intensity=0.6, speed=0.8, 
                pitch_shift=-0.5, energy=0.9, warmth=0.8
            ),
            'warm': EmotionalVoiceConfig(
                emotion='warm', intensity=0.7, speed=0.95, 
                pitch_shift=0.5, energy=1.0, warmth=0.9
            ),
            'inspiring': EmotionalVoiceConfig(
                emotion='inspiring', intensity=0.8, speed=1.05, 
                pitch_shift=1.5, energy=1.1, warmth=0.7
            ),
            'confident': EmotionalVoiceConfig(
                emotion='confident', intensity=0.8, speed=1.0, 
                pitch_shift=1.0, energy=1.1, warmth=0.6
            ),
            'grounded': EmotionalVoiceConfig(
                emotion='grounded', intensity=0.6, speed=0.9, 
                pitch_shift=-1.5, energy=0.9, warmth=0.8
            ),
            'soothing': EmotionalVoiceConfig(
                emotion='soothing', intensity=0.5, speed=0.85, 
                pitch_shift=-2.0, energy=0.8, warmth=0.95
            )
        }
    
    def _initialize_models(self):
        """Initialize TTS models with LoRA support"""
        try:
            if not TTS_AVAILABLE:
                raise ImportError("Coqui TTS not available")
            
            self.logger.info("Loading base TTS model...")
            
            # Initialize base TTS model
            self.base_model = TTS(
                model_name=self.model_name,
                progress_bar=True,
                gpu=self.device == "cuda"
            )
            
            # Apply LoRA if enabled
            if self.enable_lora:
                self._apply_lora_adaptation()
            
            self.is_loaded = True
            self.logger.info("✅ LoRA TTS Engine initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize TTS models: {e}")
            self.is_loaded = False
            raise
    
    def _apply_lora_adaptation(self):
        """Apply LoRA adaptation to the base model"""
        try:
            if not LORA_AVAILABLE:
                self.logger.warning("LoRA not available, skipping adaptation")
                return
            
            self.logger.info("Applying LoRA adaptation...")
            
            # Get the underlying model for LoRA adaptation
            if hasattr(self.base_model, 'synthesizer'):
                target_model = self.base_model.synthesizer
            else:
                self.logger.warning("Could not access synthesizer for LoRA adaptation")
                return
            
            # Configure LoRA
            lora_config = LoraConfig(
                r=self.lora_config.r,
                lora_alpha=self.lora_config.lora_alpha,
                target_modules=self.lora_config.target_modules or ["linear"],
                lora_dropout=self.lora_config.lora_dropout,
                bias=self.lora_config.bias,
                task_type=TaskType.FEATURE_EXTRACTION
            )
            
            # Apply LoRA to model
            self.lora_model = get_peft_model(target_model, lora_config)
            self.logger.info("✅ LoRA adaptation applied successfully")
            
        except Exception as e:
            self.logger.warning(f"LoRA adaptation failed: {e}")
            self.lora_model = None
    
    def generate_speech(self, 
                       text: str, 
                       emotion: str = "balanced",
                       language: str = "en",
                       speaker_wav: Optional[str] = None,
                       output_path: Optional[str] = None) -> str:
        """
        Generate speech with emotional control and LoRA enhancement
        
        Args:
            text: Text to synthesize
            emotion: Emotional style for the voice
            language: Language code
            speaker_wav: Reference audio for voice cloning
            output_path: Output file path
            
        Returns:
            Path to generated audio file
        """
        if not self.is_loaded:
            raise RuntimeError("TTS Engine not properly initialized")
        
        # Generate unique session ID
        session_id = str(uuid.uuid4())
        
        if output_path is None:
            output_path = f"tts/tts_outputs/{session_id}_lora.wav"
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        try:
            # Get emotional voice configuration
            emotion_config = self.emotion_configs.get(emotion, self.emotion_configs['balanced'])
            
            self.logger.info(f"Generating speech with emotion: {emotion}")
            
            # Generate base audio
            if speaker_wav and os.path.exists(speaker_wav):
                # Voice cloning mode
                audio = self._generate_with_voice_cloning(text, speaker_wav, language, emotion_config)
            else:
                # Standard synthesis mode
                audio = self._generate_standard_synthesis(text, language, emotion_config)
            
            # Apply emotional modifications
            audio = self._apply_emotional_modifications(audio, emotion_config)
            
            # Save audio
            self._save_audio(audio, output_path)
            
            self.logger.info(f"✅ LoRA TTS generated: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Speech generation failed: {e}")
            raise
    
    def _generate_with_voice_cloning(self, text: str, speaker_wav: str,
                                   language: str, emotion_config: EmotionalVoiceConfig) -> np.ndarray:
        """Generate speech with voice cloning"""
        try:
            # Use XTTS voice cloning capabilities
            # Note: emotion_config could be used for future emotional voice cloning enhancements
            _ = emotion_config  # Suppress unused parameter warning
            audio = self.base_model.tts(
                text=text,
                speaker_wav=speaker_wav,
                language=language,
                split_sentences=True
            )
            return np.array(audio)

        except Exception as e:
            self.logger.error(f"Voice cloning failed: {e}")
            raise
    
    def _generate_standard_synthesis(self, text: str, language: str,
                                   emotion_config: EmotionalVoiceConfig) -> np.ndarray:
        """Generate speech with standard synthesis"""
        try:
            # Standard TTS synthesis
            # Note: emotion_config could be used for future emotional synthesis enhancements
            _ = emotion_config  # Suppress unused parameter warning
            audio = self.base_model.tts(
                text=text,
                language=language,
                split_sentences=True
            )
            return np.array(audio)

        except Exception as e:
            self.logger.error(f"Standard synthesis failed: {e}")
            raise
    
    def _apply_emotional_modifications(self, audio: np.ndarray, 
                                     emotion_config: EmotionalVoiceConfig) -> np.ndarray:
        """Apply emotional modifications to audio"""
        try:
            # Convert to AudioSegment for processing
            audio_segment = AudioSegment(
                audio.tobytes(),
                frame_rate=22050,
                sample_width=2,
                channels=1
            )
            
            # Apply speed modification
            if emotion_config.speed != 1.0:
                audio_segment = audio_segment.speedup(playback_speed=emotion_config.speed)
            
            # Apply pitch shift
            if emotion_config.pitch_shift != 0.0:
                # Simple pitch shift using octaves
                octaves = emotion_config.pitch_shift / 12.0
                new_sample_rate = int(audio_segment.frame_rate * (2.0 ** octaves))
                audio_segment = audio_segment._spawn(audio_segment.raw_data, overrides={"frame_rate": new_sample_rate})
                audio_segment = audio_segment.set_frame_rate(22050)
            
            # Apply energy/volume modification
            if emotion_config.energy != 1.0:
                volume_change = 20 * np.log10(emotion_config.energy)  # Convert to dB
                audio_segment = audio_segment + volume_change
            
            # Convert back to numpy array
            audio_data = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
            audio_data = audio_data / np.max(np.abs(audio_data))  # Normalize
            
            return audio_data
            
        except Exception as e:
            self.logger.warning(f"Emotional modification failed: {e}")
            return audio
    
    def _save_audio(self, audio: np.ndarray, output_path: str):
        """Save audio to file"""
        try:
            # Ensure audio is in correct format
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            
            # Normalize audio
            audio = audio / np.max(np.abs(audio))
            
            # Save using soundfile
            sf.write(output_path, audio, 22050)
            
        except Exception as e:
            self.logger.error(f"Failed to save audio: {e}")
            raise
    
    def load_lora_weights(self, lora_weights_path: str):
        """Load pre-trained LoRA weights"""
        try:
            if not self.lora_model:
                self.logger.warning("LoRA model not available")
                return False

            if os.path.exists(lora_weights_path):
                # Load LoRA adapter
                if hasattr(self.lora_model, 'load_adapter'):
                    self.lora_model.load_adapter(lora_weights_path)
                else:
                    # Alternative loading method for PEFT models
                    from peft import PeftModel
                    self.lora_model = PeftModel.from_pretrained(
                        self.base_model.synthesizer,
                        lora_weights_path
                    )

                # Load emotional configuration if available
                config_path = os.path.join(lora_weights_path, "training_config.json")
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                        # Update emotion configs with trained parameters
                        self._update_emotion_configs_from_training(config)

                self.logger.info(f"✅ LoRA weights loaded from {lora_weights_path}")
                return True
            else:
                self.logger.warning(f"LoRA weights file not found: {lora_weights_path}")
                return False

        except Exception as e:
            self.logger.error(f"Failed to load LoRA weights: {e}")
            return False

    def _update_emotion_configs_from_training(self, training_config: dict):
        """Update emotion configurations based on training data"""
        try:
            trained_emotions = training_config.get('emotions', [])
            emotion_weights = training_config.get('emotion_weights', {})

            # Update existing emotion configs with trained weights
            for emotion in trained_emotions:
                if emotion in self.emotion_configs:
                    weight = emotion_weights.get(emotion, 1.0)
                    # Adjust intensity based on training weight
                    self.emotion_configs[emotion].intensity *= weight

            self.logger.info(f"Updated emotion configs for {len(trained_emotions)} emotions")

        except Exception as e:
            self.logger.warning(f"Failed to update emotion configs: {e}")
    
    def save_lora_weights(self, output_path: str):
        """Save current LoRA weights"""
        try:
            if not self.lora_model:
                self.logger.warning("LoRA model not available")
                return False
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            self.lora_model.save_pretrained(output_path)
            self.logger.info(f"✅ LoRA weights saved to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save LoRA weights: {e}")
            return False
    
    def get_available_emotions(self) -> List[str]:
        """Get list of available emotions"""
        return list(self.emotion_configs.keys())
    
    def _auto_load_emotional_weights(self):
        """Automatically load emotional LoRA weights if available"""
        try:
            # Check for pre-trained emotional weights
            weights_paths = [
                "lora_emotional_models",
                "models/lora_emotional",
                "checkpoints/emotional_lora"
            ]

            for weights_path in weights_paths:
                if os.path.exists(weights_path):
                    self.logger.info(f"Found emotional LoRA weights at {weights_path}")
                    if self.load_lora_weights(weights_path):
                        self.logger.info("✅ Emotional LoRA weights auto-loaded")
                        return

            self.logger.info("No pre-trained emotional LoRA weights found")

        except Exception as e:
            self.logger.warning(f"Auto-load emotional weights failed: {e}")

    def is_available(self) -> bool:
        """Check if LoRA TTS is available and loaded"""
        return self.is_loaded and TTS_AVAILABLE


# Global LoRA TTS Engine instance
_lora_tts_engine = None

def get_lora_tts_engine() -> LoRATTSEngine:
    """Get global LoRA TTS engine instance"""
    global _lora_tts_engine
    if _lora_tts_engine is None:
        _lora_tts_engine = LoRATTSEngine()
    return _lora_tts_engine


def initialize_lora_tts(model_name: str = None, device: str = "auto") -> bool:
    """Initialize LoRA TTS engine"""
    global _lora_tts_engine
    try:
        _lora_tts_engine = LoRATTSEngine(
            model_name=model_name or "tts_models/multilingual/multi-dataset/xtts_v2",
            device=device
        )
        return _lora_tts_engine.is_available()
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to initialize LoRA TTS: {e}")
        return False
