from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from gtts import gTTS
import uuid
import subprocess
import os
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
import traceback
from keras.models import load_model
import json
from datetime import datetime, timezone
import logging
import sys
import os
from pydub import AudioSegment
from pydub.effects import normalize
import tempfile
import io
import boto3
from botocore.exceptions import ClientError
import asyncio
import aiofiles
from typing import Optional, Dict, Any
from lesson_manager import lesson_manager
from sync_map_generator import sync_map_generator

# LoRA TTS Integration
try:
    from lora_tts_engine import LoRATTSEngine, get_lora_tts_engine, initialize_lora_tts
    LORA_TTS_AVAILABLE = True
    print("✅ LoRA TTS Engine available")
except ImportError as e:
    print(f"⚠️ LoRA TTS Engine not available: {e}")
    print("   Install dependencies: pip install -r requirements_lora_tts.txt")
    LORA_TTS_AVAILABLE = False

# Emotional Fallback TTS Integration
try:
    from emotional_fallback_tts import get_emotional_fallback_tts
    EMOTIONAL_FALLBACK_AVAILABLE = True
    print("✅ Emotional Fallback TTS available")
except ImportError as e:
    print(f"⚠️ Emotional Fallback TTS not available: {e}")
    EMOTIONAL_FALLBACK_AVAILABLE = False

# Set environment variables to handle Unicode properly
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['PYTHONUTF8'] = '1'

# Configure safe logging without Unicode characters
class SafeFormatter(logging.Formatter):
    """Custom formatter that replaces Unicode characters with ASCII equivalents"""

    EMOJI_REPLACEMENTS = {
        '🚀': '[ROCKET]',
        '✅': '[OK]',
        '❌': '[ERROR]',
        '⚠️': '[WARNING]',
        '🎵': '[MUSIC]',
        '🔔': '[BELL]',
        '🎯': '[TARGET]',
        '🧠': '[BRAIN]',
        '💻': '[COMPUTER]',
        '📊': '[CHART]',
        '🖥️': '[DESKTOP]',
        '💾': '[DISK]',
        '📋': '[CLIPBOARD]',
        '📝': '[MEMO]',
        '😊': '[SMILE]',
        '🎧': '[HEADPHONES]',
        '🖼️': '[PICTURE]',
        '🔄': '[REFRESH]',
        '🔧': '[WRENCH]',
        '🧪': '[TEST]',
        '🎉': '[PARTY]',
        '🔥': '[FIRE]',
        '⭐': '[STAR]',
        '💡': '[BULB]',
        '🎨': '[ART]',
        '📈': '[TRENDING_UP]',
        '🎪': '[CIRCUS]',
        '🌟': '[GLOWING_STAR]',
        '🎭': '[MASKS]',
        '🎬': '[CLAPPER]',
        '🎤': '[MIC]',
        '🎸': '[GUITAR]',
        '🎹': '[PIANO]',
        '🎺': '[TRUMPET]',
        '🥁': '[DRUM]',
        '🎻': '[VIOLIN]'
    }

    def format(self, record):
        # Get the original formatted message
        msg = super().format(record)

        # Replace Unicode characters with ASCII equivalents
        for emoji, replacement in self.EMOJI_REPLACEMENTS.items():
            msg = msg.replace(emoji, replacement)

        return msg

# Configure logging with safe formatter
safe_formatter = SafeFormatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(safe_formatter)

logging.basicConfig(
    level=logging.INFO,
    handlers=[console_handler]
)

from translation_agent import translate_text_with_gemini, LANGUAGE_MAP  # Added LANGUAGE_MAP

# Setup Multimodal Sentiment Analysis Integration
try:
    import sys
    import os

    # Add the current directory to Python path to enable relative imports
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)

    # Import from the multimodal_sentiment package
    from multimodal_sentiment.integration_hooks import (
        get_sentiment_from_user_profile,
        get_tts_emotion_for_script,
        health_check,
        initialize_hooks
    )

    # Initialize hooks with correct paths relative to the multimodal_sentiment directory
    multimodal_dir = os.path.join(current_dir, 'multimodal_sentiment')
    initialize_hooks(
        model_path=os.path.join(multimodal_dir, 'checkpoints', 'model_weights.pth'),
        config_path=os.path.join(multimodal_dir, 'checkpoints', 'training_config.json'),
        encoders_path=os.path.join(multimodal_dir, 'checkpoints', 'label_encoders.pkl')
    )

    print("✅ Multimodal sentiment analysis integration loaded")
    SENTIMENT_INTEGRATION_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Sentiment analysis integration not available: {e}")
    SENTIMENT_INTEGRATION_AVAILABLE = False

app = FastAPI()

# Directories
TTS_OUTPUT_DIR = "tts/tts_outputs"
RESULTS_DIR = "results"
AVATAR_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "avatars"))
GENDER_MODEL_PATH = "gender-recognition-by-voice/results/model.h5"
WAV2LIP_PATH = "Wav2Lip"
WAV2LIP_CHECKPOINT = "checkpoints/wav2lip_gan.pth"
TRANSITION_SOUNDS_DIR = "transition_sounds"

os.makedirs(TTS_OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(TRANSITION_SOUNDS_DIR, exist_ok=True)

# Enhanced TTS Configuration
class TTSConfig:
    """Configuration for TTS with fallback pipeline and audio optimization"""

    def __init__(self):
        # Audio optimization settings
        self.enable_audio_compression = True
        self.enable_transition_tones = True
        self.compression_quality = 128  # kbps for MP3
        self.target_sample_rate = 22050  # Standard for TTS

        # Emotion-based transition tones
        self.transition_tones = {
            'joyful': {'freq': 800, 'duration': 300, 'fade': 50},
            'enthusiastic': {'freq': 1000, 'duration': 250, 'fade': 30},
            'peaceful': {'freq': 400, 'duration': 500, 'fade': 100},
            'balanced': {'freq': 600, 'duration': 200, 'fade': 50},
            'contemplative': {'freq': 300, 'duration': 600, 'fade': 150},
            'warm': {'freq': 500, 'duration': 400, 'fade': 80},
            'confident': {'freq': 900, 'duration': 300, 'fade': 40},
            'soothing': {'freq': 350, 'duration': 450, 'fade': 120},
            'inspiring': {'freq': 1200, 'duration': 350, 'fade': 60},
            'grounded': {'freq': 450, 'duration': 400, 'fade': 90}
        }

# Cloud Storage Manager
class CloudStorageManager:
    """Manages cloud storage operations for TTS assets"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.s3_client = None
        self.bucket_name = os.getenv('AWS_S3_BUCKET', 'tts-assets-bucket')
        self.cdn_base_url = os.getenv('CDN_BASE_URL', f'https://{self.bucket_name}.s3.amazonaws.com')

        # Initialize S3 client if credentials are available
        try:
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                region_name=os.getenv('AWS_REGION', 'us-east-1')
            )
            self.logger.info("✅ AWS S3 client initialized")
        except Exception as e:
            self.logger.warning(f"⚠️ S3 client initialization failed: {e}")

    async def upload_asset(self, local_path: str, s3_key: str, content_type: str = 'application/octet-stream') -> Optional[str]:
        """Upload asset to S3 and return public URL"""
        if not self.s3_client:
            self.logger.warning("S3 client not available, skipping upload")
            return None

        try:
            # Upload file to S3
            self.s3_client.upload_file(
                local_path,
                self.bucket_name,
                s3_key,
                ExtraArgs={
                    'ContentType': content_type,
                    'ACL': 'public-read'
                }
            )

            # Return public URL
            public_url = f"{self.cdn_base_url}/{s3_key}"
            self.logger.info(f"✅ Uploaded {local_path} to {public_url}")
            return public_url

        except ClientError as e:
            self.logger.error(f"❌ S3 upload failed: {e}")
            return None

    async def upload_lesson_assets(self, session_id: str, audio_path: str, video_path: str, metadata: Dict[str, Any]) -> Dict[str, str]:
        """Upload complete lesson assets and return URLs"""
        urls = {}

        # Upload audio file
        audio_key = f"lessons/audio/{session_id}.mp3"
        audio_url = await self.upload_asset(audio_path, audio_key, 'audio/mpeg')
        if audio_url:
            urls['audio_url'] = audio_url

        # Upload video file
        video_key = f"lessons/video/{session_id}.mp4"
        video_url = await self.upload_asset(video_path, video_key, 'video/mp4')
        if video_url:
            urls['video_url'] = video_url

        # Upload metadata as JSON
        metadata_key = f"lessons/metadata/{session_id}.json"
        metadata_path = f"temp_metadata_{session_id}.json"

        # Save metadata temporarily
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        metadata_url = await self.upload_asset(metadata_path, metadata_key, 'application/json')
        if metadata_url:
            urls['metadata_url'] = metadata_url

        # Clean up temp file
        if os.path.exists(metadata_path):
            os.remove(metadata_path)

        return urls

# Enhanced TTS Engine with Fallback Pipeline
class EnhancedTTSEngine:
    """TTS engine with LoRA fallback, audio compression, and transition tones"""

    def __init__(self, config: TTSConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def generate_speech(self, text: str, emotion: str = 'balanced', lang: str = 'en') -> str:
        """
        Generate speech with fallback pipeline and audio optimization

        Args:
            text: Text to convert to speech
            emotion: Emotion for transition tone selection
            lang: Language code

        Returns:
            Path to optimized audio file
        """
        session_id = str(uuid.uuid4())

        try:
            # Primary: Try LoRA-enhanced TTS (placeholder for future LoRA integration)
            return self._generate_with_lora_fallback(text, emotion, lang, session_id)

        except Exception as e:
            self.logger.warning(f"LoRA TTS failed: {e}, falling back to basic gTTS")

            try:
                # Fallback: Basic gTTS with optimizations
                return self._generate_basic_optimized_tts(text, emotion, lang, session_id)

            except Exception as e2:
                self.logger.error(f"All TTS methods failed: {e2}")
                raise HTTPException(status_code=500, detail=f"TTS generation failed: {e2}")

    def _generate_with_lora_fallback(self, text: str, emotion: str, lang: str, session_id: str) -> str:
        """Primary TTS with LoRA implementation"""

        if not LORA_TTS_AVAILABLE:
            raise Exception("LoRA TTS not available, falling back to gTTS")

        try:
            # Initialize LoRA TTS engine if not already done
            lora_engine = get_lora_tts_engine()

            if not lora_engine.is_available():
                raise Exception("LoRA TTS engine not properly initialized")

            # Generate speech with LoRA TTS
            output_path = os.path.join(TTS_OUTPUT_DIR, f"{session_id}_lora.wav")

            self.logger.info(f"[LORA] Generating speech with emotion: {emotion}")

            # Use LoRA TTS with emotional control
            lora_audio_path = lora_engine.generate_speech(
                text=text,
                emotion=emotion,
                language=lang,
                output_path=output_path
            )

            # Convert to MP3 for consistency with the rest of the pipeline
            mp3_path = os.path.join(TTS_OUTPUT_DIR, f"{session_id}_lora.mp3")
            self._convert_wav_to_mp3(lora_audio_path, mp3_path)

            # Apply additional optimizations
            optimized_path = self._optimize_audio(mp3_path, emotion, session_id)

            self.logger.info(f"[OK] LoRA TTS generated successfully: {optimized_path}")
            return optimized_path

        except Exception as e:
            self.logger.warning(f"LoRA TTS generation failed: {e}")
            raise Exception(f"LoRA TTS failed: {e}")

    def _generate_basic_optimized_tts(self, text: str, emotion: str, lang: str, session_id: str) -> str:
        """Generate optimized TTS with enhanced emotional processing"""

        try:
            # Try enhanced emotional fallback first
            if EMOTIONAL_FALLBACK_AVAILABLE:
                self.logger.info(f"[EMOTION] Using enhanced emotional fallback for: {emotion}")

                emotional_tts = get_emotional_fallback_tts()
                emotional_path = emotional_tts.generate_emotional_speech(
                    text=text,
                    emotion=emotion,
                    language=lang
                )

                # Apply additional optimizations
                optimized_path = self._optimize_audio(emotional_path, emotion, session_id)

                self.logger.info(f"[OK] Enhanced emotional TTS generated: {optimized_path}")
                return optimized_path

        except Exception as e:
            self.logger.warning(f"Enhanced emotional fallback failed: {e}")

        # Fallback to basic gTTS with standard optimizations
        self.logger.info("[FALLBACK] Using basic gTTS with standard optimizations")

        # Generate base audio with gTTS
        base_mp3_path = os.path.join(TTS_OUTPUT_DIR, f"{session_id}_base.mp3")
        tts = gTTS(text=text, lang=lang, slow=False)
        tts.save(base_mp3_path)

        self.logger.info(f"[OK] Basic gTTS generated: {base_mp3_path}")

        # Apply audio optimizations
        optimized_path = self._optimize_audio(base_mp3_path, emotion, session_id)

        return optimized_path

    def _optimize_audio(self, audio_path: str, emotion: str, session_id: str) -> str:
        """Apply audio compression and add transition tones"""

        try:
            # Load audio with pydub
            audio = AudioSegment.from_mp3(audio_path)

            # Normalize audio levels
            audio = normalize(audio)

            # Resample to target sample rate for consistency
            if audio.frame_rate != self.config.target_sample_rate:
                audio = audio.set_frame_rate(self.config.target_sample_rate)

            # Convert to mono for smaller file size
            if audio.channels > 1:
                audio = audio.set_channels(1)

            # Add transition tone if enabled
            if self.config.enable_transition_tones:
                audio = self._add_transition_tone(audio, emotion)

            # Apply compression and save optimized audio
            optimized_path = os.path.join(TTS_OUTPUT_DIR, f"{session_id}_optimized.mp3")

            if self.config.enable_audio_compression:
                audio.export(
                    optimized_path,
                    format="mp3",
                    bitrate=f"{self.config.compression_quality}k",
                    parameters=["-ac", "1"]  # Force mono
                )
            else:
                audio.export(optimized_path, format="mp3")

            self.logger.info(f"[MUSIC] Audio optimized: {optimized_path}")
            return optimized_path

        except Exception as e:
            self.logger.warning(f"Audio optimization failed: {e}, using original")
            return audio_path

    def _add_transition_tone(self, audio: AudioSegment, emotion: str) -> AudioSegment:
        """Add emotion-based transition tone before the main audio"""

        try:
            # Get tone parameters for the emotion
            tone_params = self.config.transition_tones.get(emotion, self.config.transition_tones['balanced'])

            # Generate synthetic tone
            tone = self._generate_synthetic_tone(tone_params)

            if tone:
                # Add a small gap between tone and speech
                gap = AudioSegment.silent(duration=200)  # 200ms gap

                # Combine: tone + gap + main audio
                combined = tone + gap + audio

                self.logger.info(f"[BELL] Added transition tone for emotion: {emotion}")
                return combined

        except Exception as e:
            self.logger.warning(f"Transition tone addition failed: {e}")

        return audio

    def _generate_synthetic_tone(self, tone_params: dict) -> AudioSegment:
        """Generate synthetic transition tone"""

        try:
            # Generate sine wave tone using numpy
            sample_rate = self.config.target_sample_rate
            duration_seconds = tone_params['duration'] / 1000.0
            t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds), False)
            wave = np.sin(2 * np.pi * tone_params['freq'] * t)

            # Convert to 16-bit PCM
            wave = (wave * 16383).astype(np.int16)  # Reduced amplitude for subtlety

            # Create AudioSegment from numpy array
            tone = AudioSegment(
                wave.tobytes(),
                frame_rate=sample_rate,
                sample_width=2,
                channels=1
            )

            # Apply fade in/out
            tone = tone.fade_in(tone_params['fade']).fade_out(tone_params['fade'])

            # Reduce volume to be subtle
            tone = tone - 12  # Reduce by 12dB

            return tone

        except Exception as e:
            self.logger.warning(f"Synthetic tone generation failed: {e}")
            return AudioSegment.silent(duration=100)  # Return short silence as fallback

    def _convert_wav_to_mp3(self, wav_path: str, mp3_path: str):
        """Convert WAV file to MP3"""
        try:
            audio = AudioSegment.from_wav(wav_path)
            audio.export(mp3_path, format="mp3", bitrate=f"{self.config.compression_quality}k")
            self.logger.info(f"[OK] Converted WAV to MP3: {mp3_path}")
        except Exception as e:
            self.logger.error(f"WAV to MP3 conversion failed: {e}")
            raise

# Global TTS configuration and engine
tts_config = TTSConfig()
enhanced_tts = EnhancedTTSEngine(tts_config)
cloud_storage = CloudStorageManager()

# Initialize LoRA TTS Engine
if LORA_TTS_AVAILABLE:
    try:
        print("🚀 Initializing LoRA TTS Engine...")
        lora_initialized = initialize_lora_tts()
        if lora_initialized:
            print("✅ LoRA TTS Engine initialized successfully")
        else:
            print("⚠️ LoRA TTS Engine initialization failed")
            LORA_TTS_AVAILABLE = False
    except Exception as e:
        print(f"❌ LoRA TTS Engine initialization error: {e}")
        LORA_TTS_AVAILABLE = False






print(f"Avatar directory: {AVATAR_DIR}")

gender_model = load_model(GENDER_MODEL_PATH) if os.path.exists(GENDER_MODEL_PATH) else None

AVATARS = {
    "female": os.path.join(AVATAR_DIR, "pht2.jpg"),
    "default": os.path.join(AVATAR_DIR, "pht2.jpg")
}

for gender, path in AVATARS.items():
    if not os.path.isfile(path):
        print(f"⚠️ Missing avatar file: {path}")

def extract_features(file_path: str) -> np.ndarray:
    try:
        from scipy.io import wavfile
        sample_rate, X = wavfile.read(file_path)

        if X.ndim > 1:
            X = X[:, 0]

        X = X.astype(np.float32)
        X = X / np.max(np.abs(X), axis=0)

        fft_spectrum = np.fft.fft(X)
        magnitude = np.abs(fft_spectrum[:len(fft_spectrum)//2])
        mel = np.log1p(magnitude[:128])

        if mel.size < 128:
            mel = np.pad(mel, (0, 128 - mel.size), mode='constant')

        return mel
    except Exception:
        print("Feature extraction error:", traceback.format_exc())
        return np.array([])

def predict_gender(audio_path: str) -> str:
    if gender_model is None:
        return "default"

    features = extract_features(audio_path)
    if features.size != 128:
        features = np.pad(features, (0, 128 - features.shape[0]), mode='constant')

    features = np.expand_dims(features, axis=0)
    prediction = gender_model.predict(features, verbose=0)[0]
    return "male" if prediction >= 0.5 else "female"

def select_avatar(gender: str = "female") -> str:
    # Always use female avatars since TTS voice is female
    avatar_path = AVATARS["female"]
    if not os.path.isfile(avatar_path):
        raise FileNotFoundError(f"Avatar file not found: {avatar_path}")
    print(f"Selected female avatar path: {avatar_path}")
    return avatar_path

def analyze_text_sentiment(text: str, user_persona: str = "youth", user_id: str = None) -> dict:
    """
    Analyze text sentiment using multimodal sentiment analysis integration.

    Args:
        text: Text to analyze
        user_persona: User persona ('youth', 'kids', 'spiritual')
        user_id: Optional user identifier

    Returns:
        Dictionary containing sentiment analysis results and TTS emotion mapping
    """
    if not SENTIMENT_INTEGRATION_AVAILABLE:
        # Fallback to simple sentiment analysis
        return {
            'sentiment': 'neutral',
            'confidence': 0.6,
            'tts_emotion': 'balanced',
            'persona': user_persona,
            'analysis_type': 'fallback'
        }

    try:
        # Get sentiment analysis from integration hooks
        sentiment_result = get_sentiment_from_user_profile(
            text=text,
            user_persona=user_persona,
            user_id=user_id
        )

        # Get TTS emotion mapping for the text
        tts_result = get_tts_emotion_for_script(
            script_text=text,
            target_persona=user_persona
        )

        # Combine results
        return {
            'sentiment': sentiment_result.get('sentiment', 'neutral'),
            'confidence': sentiment_result.get('confidence', 0.6),
            'tts_emotion': sentiment_result.get('tts_emotion', 'balanced'),
            'persona': user_persona,
            'analysis_type': sentiment_result.get('analysis_type', 'multimodal'),
            'tone': sentiment_result.get('tone', 'neutral'),
            'primary_emotion': tts_result.get('primary_emotion', 'balanced'),
            'emotion_intensity': tts_result.get('emotion_intensity', 0.5),
            'tts_parameters': tts_result.get('tts_parameters', {}),
            'voice_recommendations': tts_result.get('voice_recommendations', {}),
            'recommendations': sentiment_result.get('recommendations', {})
        }

    except Exception as e:
        print(f"Error in sentiment analysis: {e}")
        return {
            'sentiment': 'neutral',
            'confidence': 0.6,
            'tts_emotion': 'balanced',
            'persona': user_persona,
            'analysis_type': 'error_fallback',
            'error': str(e)
        }

def convert_mp3_to_wav(mp3_path: str, wav_path: str):
    """Convert MP3 to WAV using librosa (most reliable method)."""
    try:
        print(f"Converting {mp3_path} to {wav_path}...")

        # Check if input file exists
        if not os.path.exists(mp3_path):
            raise FileNotFoundError(f"Input MP3 file not found: {mp3_path}")

        # Ensure output directory exists
        os.makedirs(os.path.dirname(wav_path), exist_ok=True)

        # Use librosa to load the audio file (it can handle MP3 via audioread)
        # Set sr=None to preserve original sample rate
        audio_data, sample_rate = librosa.load(mp3_path, sr=22050)  # Use standard sample rate
        print(f"Loaded audio: {len(audio_data)} samples at {sample_rate} Hz")

        # Save as WAV using soundfile
        sf.write(wav_path, audio_data, sample_rate)
        print(f"Successfully converted to WAV: {wav_path}")

        # Verify the output file was created and has content
        if not os.path.exists(wav_path) or os.path.getsize(wav_path) == 0:
            raise Exception("WAV file was not created successfully or is empty")

    except Exception as e:
        print(f"Error in audio conversion: {e}")
        import traceback
        traceback.print_exc()
        raise Exception(f"Audio conversion failed: {e}")

def run_wav2lip(audio_path: str, image_path: str, output_path: str):
    subprocess.run([
        "python", "inference.py",
        "--checkpoint_path", WAV2LIP_CHECKPOINT,
        "--face", os.path.abspath(image_path),
        "--audio", os.path.abspath(audio_path),
        "--outfile", os.path.abspath(output_path)
    ], cwd=WAV2LIP_PATH, check=True)

def generate_video_metadata(session_id: str, text: str, language: str, gender: str, avatar_path: str,
                          sentiment_analysis: dict = None, user_persona: str = 'youth') -> dict:
    """Generate metadata for the created video.
    
    Args:
        session_id: Unique identifier for the session
        text: The text used for TTS
        language: Language code
        gender: Detected gender for avatar selection
        avatar_path: Path to the avatar image used
        
    Returns:
        Dictionary containing video metadata
    """
    lang_name, script = LANGUAGE_MAP.get(language, ('Unknown', 'Latin'))
    
    metadata = {
        'session_id': session_id,
        'language': language,
        'language_name': lang_name,
        'script': script,
        'text_length': len(text),
        'gender': gender,
        'avatar': os.path.basename(avatar_path),
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'video_format': 'mp4',
        'user_persona': user_persona
    }

    # Add sentiment analysis data if available
    if sentiment_analysis:
        metadata['sentiment_analysis'] = {
            'sentiment': sentiment_analysis.get('sentiment', 'neutral'),
            'confidence': sentiment_analysis.get('confidence', 0.6),
            'tts_emotion': sentiment_analysis.get('tts_emotion', 'balanced'),
            'tone': sentiment_analysis.get('tone', 'neutral'),
            'analysis_type': sentiment_analysis.get('analysis_type', 'unknown'),
            'emotion_intensity': sentiment_analysis.get('emotion_intensity', 0.5)
        }

        # Add TTS parameters if available
        if 'tts_parameters' in sentiment_analysis:
            metadata['tts_parameters'] = sentiment_analysis['tts_parameters']

        # Add voice recommendations if available
        if 'voice_recommendations' in sentiment_analysis:
            metadata['voice_recommendations'] = sentiment_analysis['voice_recommendations']
    
    # Save metadata to file
    metadata_path = os.path.join(RESULTS_DIR, f"metadata_{session_id}.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"Generated metadata for session: {session_id}")
    return metadata

@app.post("/api/generate-and-sync")
async def generate_and_sync(
    text: str = Form(...),
    target_lang: str = Form(default='en'),
    user_persona: str = Form(default='youth'),
    user_id: str = Form(default=None)
):
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")

    if len(text) > 500:
        text = text[:500]

    original_text = text
    if target_lang != "en":
        translated_text, confidence = translate_text_with_gemini(text, target_lang, source_lang='en')
        if not translated_text or confidence < 0.1:
            raise HTTPException(status_code=500, detail=f"Translation failed: {translated_text}")
        text = translated_text

    # Validate user persona
    if user_persona not in ['youth', 'kids', 'spiritual']:
        user_persona = 'youth'  # Default fallback

    # Perform sentiment analysis on the text
    sentiment_analysis = analyze_text_sentiment(
        text=text,
        user_persona=user_persona,
        user_id=user_id
    )

    print(f"Sentiment Analysis: {sentiment_analysis['sentiment']} "
          f"(confidence: {sentiment_analysis['confidence']:.3f}, "
          f"emotion: {sentiment_analysis['tts_emotion']})")

    session_id = str(uuid.uuid4())

    try:
        # Get TTS emotion from sentiment analysis
        tts_emotion = sentiment_analysis.get('tts_emotion', 'balanced')
        print(f"[MUSIC] Generating TTS with emotion: {tts_emotion}")

        # Use enhanced TTS engine with fallback pipeline
        optimized_audio_path = enhanced_tts.generate_speech(
            text=text,
            emotion=tts_emotion,
            lang='en'
        )

        print(f"[OK] Enhanced TTS with fallback generated: {optimized_audio_path}")

        # Convert to WAV for Wav2Lip
        wav_path = os.path.join(TTS_OUTPUT_DIR, f"{session_id}.wav")
        convert_mp3_to_wav(optimized_audio_path, wav_path)

        # Always use female avatars since TTS voice is female
        gender = "female"
        avatar_path = select_avatar(gender)

        output_video = os.path.join(RESULTS_DIR, f"{session_id}.mp4")
        run_wav2lip(wav_path, avatar_path, output_video)
        
        # Generate and save metadata
        metadata = generate_video_metadata(
            session_id=session_id,
            text=original_text if target_lang == 'en' else text,
            language=target_lang,
            gender=gender,
            avatar_path=avatar_path,
            sentiment_analysis=sentiment_analysis,
            user_persona=user_persona
        )

        # Generate sync map for UI controls and visual sync
        try:
            sync_map = sync_map_generator.generate_sync_map(
                audio_path=optimized_audio_path,
                text=original_text if target_lang == 'en' else text,
                session_id=session_id
            )

            # Save sync map
            sync_map_path = sync_map_generator.save_sync_map(sync_map)

            # Add sync map to metadata
            metadata['sync_map'] = sync_map
            metadata['sync_map_path'] = sync_map_path

            print(f"✅ Generated sync map with {len(sync_map.get('word_timestamps', []))} word timestamps")

        except Exception as e:
            print(f"⚠️ Sync map generation failed: {e}")
            # Continue without sync map

        # Upload assets to cloud storage
        try:
            cloud_urls = await cloud_storage.upload_lesson_assets(
                session_id=session_id,
                audio_path=optimized_audio_path,
                video_path=output_video,
                metadata=metadata
            )

            # Add cloud URLs to metadata
            metadata.update(cloud_urls)

            # Save updated metadata with cloud URLs
            metadata_path = os.path.join(RESULTS_DIR, f"metadata_{session_id}.json")
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

            print(f"✅ Assets uploaded to cloud: {cloud_urls}")

        except Exception as e:
            print(f"⚠️ Cloud upload failed: {e}")
            # Continue without cloud URLs

        return FileResponse(
            path=output_video,
            filename=f"lipsync_{session_id}.mp4",
            media_type="video/mp4"
        )

    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"FFmpeg/Wav2Lip failed: {e.stderr}")
    except Exception:
        print("Unexpected error:", traceback.format_exc())
        raise HTTPException(status_code=500, detail="Unexpected error occurred")

@app.get("/")
def root():
    return {"message": "TTS-LipSync-Translation API running"}

@app.get("/api/metadata/{session_id}")
async def get_metadata(session_id: str):
    """Retrieve metadata for a specific video session."""
    metadata_path = os.path.join(RESULTS_DIR, f"metadata_{session_id}.json")

    if not os.path.exists(metadata_path):
        raise HTTPException(status_code=404, detail="Metadata not found")

    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        return JSONResponse(content=metadata)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load metadata: {str(e)}")

@app.post("/api/analyze-sentiment")
async def analyze_sentiment_endpoint(
    text: str = Form(...),
    user_persona: str = Form(default='youth'),
    user_id: str = Form(default=None)
):
    """Analyze text sentiment without generating video."""
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")

    try:
        sentiment_result = analyze_text_sentiment(
            text=text,
            user_persona=user_persona,
            user_id=user_id
        )
        return JSONResponse(content=sentiment_result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sentiment analysis failed: {str(e)}")

@app.get("/api/sentiment-health")
async def sentiment_health_check():
    """Check the health of sentiment analysis integration."""
    if not SENTIMENT_INTEGRATION_AVAILABLE:
        return JSONResponse(content={
            'status': 'unavailable',
            'message': 'Sentiment analysis integration not loaded'
        })

    try:
        health_result = health_check()
        return JSONResponse(content=health_result)
    except Exception as e:
        return JSONResponse(content={
            'status': 'error',
            'message': f'Health check failed: {str(e)}'
        })

# Enhanced TTS Configuration Endpoints
@app.get("/api/tts-config")
async def get_tts_config():
    """Get current TTS configuration"""
    return {
        "audio_compression_enabled": tts_config.enable_audio_compression,
        "transition_tones_enabled": tts_config.enable_transition_tones,
        "compression_quality": tts_config.compression_quality,
        "target_sample_rate": tts_config.target_sample_rate,
        "available_emotions": list(tts_config.transition_tones.keys()),
        "transition_tone_params": tts_config.transition_tones
    }

@app.post("/api/tts-config/toggle")
async def toggle_tts_feature(
    feature: str = Form(...),
    enabled: bool = Form(...)
):
    """Toggle TTS features on/off"""

    if feature == "audio_compression":
        tts_config.enable_audio_compression = enabled
    elif feature == "transition_tones":
        tts_config.enable_transition_tones = enabled
    else:
        raise HTTPException(status_code=400, detail=f"Unknown feature: {feature}")

    return {
        "message": f"Feature '{feature}' {'enabled' if enabled else 'disabled'}",
        "current_config": {
            "audio_compression_enabled": tts_config.enable_audio_compression,
            "transition_tones_enabled": tts_config.enable_transition_tones
        }
    }

@app.post("/api/tts-config/compression-quality")
async def set_compression_quality(quality: int = Form(...)):
    """Set audio compression quality (64-320 kbps)"""

    if not 64 <= quality <= 320:
        raise HTTPException(status_code=400, detail="Quality must be between 64 and 320 kbps")

    tts_config.compression_quality = quality

    return {
        "message": f"Compression quality set to {quality} kbps",
        "compression_quality": tts_config.compression_quality
    }

# Lesson Management API Endpoints
@app.post("/api/lessons/create")
async def create_lesson(
    title: str = Form(...),
    content: str = Form(...),
    category: str = Form(default="general"),
    target_age: str = Form(default="elementary"),
    difficulty: str = Form(default="beginner"),
    duration_minutes: int = Form(default=5)
):
    """Create a new lesson"""
    try:
        lesson_id = lesson_manager.create_lesson(
            title=title,
            content=content,
            category=category,
            target_age=target_age,
            difficulty=difficulty,
            duration_minutes=duration_minutes
        )

        return {
            "status": "success",
            "lesson_id": lesson_id,
            "message": f"Lesson '{title}' created successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create lesson: {str(e)}")

@app.get("/api/lessons/{lesson_id}")
async def get_lesson(lesson_id: str):
    """Get lesson data by ID"""
    lesson_data = lesson_manager.get_lesson(lesson_id)

    if not lesson_data:
        raise HTTPException(status_code=404, detail="Lesson not found")

    return lesson_data

@app.get("/api/lessons")
async def get_lessons_index():
    """Get complete lessons index"""
    return lesson_manager.get_lessons_index()

@app.get("/api/lessons/category/{category}")
async def get_lessons_by_category(category: str):
    """Get all lessons in a category"""
    lessons = lesson_manager.get_lessons_by_category(category)
    return {
        "category": category,
        "total_lessons": len(lessons),
        "lessons": lessons
    }

@app.post("/api/lessons/{lesson_id}/generate-assets")
async def generate_lesson_assets(lesson_id: str):
    """Generate TTS assets for a specific lesson"""
    lesson_data = lesson_manager.get_lesson(lesson_id)

    if not lesson_data:
        raise HTTPException(status_code=404, detail="Lesson not found")

    try:
        # Generate TTS assets using the lesson content
        session_id = str(uuid.uuid4())
        content = lesson_data["content"]
        tts_emotion = lesson_data["tts_config"]["emotion"]

        # Use enhanced TTS engine
        optimized_audio_path = enhanced_tts.generate_speech(
            text=content,
            emotion=tts_emotion,
            lang='en'
        )

        # Convert to WAV for Wav2Lip
        wav_path = os.path.join(TTS_OUTPUT_DIR, f"{session_id}.wav")
        convert_mp3_to_wav(optimized_audio_path, wav_path)

        # Generate video with female avatar
        avatar_path = select_avatar("female")
        output_video = os.path.join(RESULTS_DIR, f"{session_id}.mp4")
        run_wav2lip(wav_path, avatar_path, output_video)

        # Generate metadata
        metadata = generate_video_metadata(
            session_id=session_id,
            text=content,
            language="en",
            gender="female",
            avatar_path=avatar_path,
            sentiment_analysis={"sentiment": "neutral", "confidence": 0.8, "tts_emotion": tts_emotion},
            user_persona="educational"
        )

        # Generate sync map for lesson
        try:
            sync_map = sync_map_generator.generate_sync_map(
                audio_path=optimized_audio_path,
                text=content,
                session_id=session_id
            )

            # Save sync map
            sync_map_path = sync_map_generator.save_sync_map(sync_map)
            metadata['sync_map'] = sync_map
            metadata['sync_map_path'] = sync_map_path

        except Exception as e:
            print(f"⚠️ Lesson sync map generation failed: {e}")

        # Upload to cloud storage
        cloud_urls = await cloud_storage.upload_lesson_assets(
            session_id=session_id,
            audio_path=optimized_audio_path,
            video_path=output_video,
            metadata=metadata
        )

        # Update lesson with assets
        lesson_manager.update_lesson_assets(lesson_id, cloud_urls)

        return {
            "status": "success",
            "lesson_id": lesson_id,
            "session_id": session_id,
            "assets": cloud_urls,
            "message": "Assets generated and uploaded successfully"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Asset generation failed: {str(e)}")

@app.post("/api/lessons/create-samples")
async def create_sample_lessons():
    """Create 4 sample lessons for team testing"""
    try:
        lesson_ids = lesson_manager.create_sample_lessons()
        return {
            "status": "success",
            "created_lessons": lesson_ids,
            "total_created": len(lesson_ids),
            "message": "Sample lessons created successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create sample lessons: {str(e)}")

# Sync Map API Endpoints
@app.get("/api/sync-map/{session_id}")
async def get_sync_map(session_id: str):
    """Get sync map data for a session"""
    sync_map_path = f"sync_maps/sync_map_{session_id}.json"

    if not os.path.exists(sync_map_path):
        raise HTTPException(status_code=404, detail="Sync map not found")

    try:
        with open(sync_map_path, 'r') as f:
            sync_map = json.load(f)
        return sync_map
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load sync map: {str(e)}")

@app.post("/api/generate-sync-map")
async def generate_sync_map_endpoint(
    audio_path: str = Form(...),
    text: str = Form(...),
    session_id: str = Form(default=None)
):
    """Generate sync map for existing audio file"""

    if not session_id:
        session_id = str(uuid.uuid4())

    if not os.path.exists(audio_path):
        raise HTTPException(status_code=404, detail="Audio file not found")

    try:
        sync_map = sync_map_generator.generate_sync_map(
            audio_path=audio_path,
            text=text,
            session_id=session_id
        )

        sync_map_path = sync_map_generator.save_sync_map(sync_map)

        return {
            "status": "success",
            "session_id": session_id,
            "sync_map_path": sync_map_path,
            "sync_map": sync_map,
            "message": "Sync map generated successfully"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sync map generation failed: {str(e)}")

# Asset Fetch API Endpoints for Vedant
@app.get("/api/assets/{session_id}")
async def get_session_assets(session_id: str):
    """Get all assets for a session (audio, video, metadata, sync map)"""

    # Check if session exists
    metadata_path = os.path.join(RESULTS_DIR, f"metadata_{session_id}.json")
    if not os.path.exists(metadata_path):
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        # Load metadata
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        # Collect all asset paths
        assets = {
            "session_id": session_id,
            "metadata": metadata,
            "local_paths": {
                "video": os.path.join(RESULTS_DIR, f"{session_id}.mp4"),
                "audio": None,  # Will be found dynamically
                "sync_map": f"sync_maps/sync_map_{session_id}.json"
            },
            "cloud_urls": {
                "video_url": metadata.get("video_url"),
                "audio_url": metadata.get("audio_url"),
                "metadata_url": metadata.get("metadata_url")
            },
            "status": {
                "video_exists": os.path.exists(os.path.join(RESULTS_DIR, f"{session_id}.mp4")),
                "sync_map_exists": os.path.exists(f"sync_maps/sync_map_{session_id}.json"),
                "cloud_uploaded": bool(metadata.get("video_url"))
            }
        }

        # Find audio file (could be optimized.mp3 or base.mp3)
        for audio_suffix in ["_optimized.mp3", "_enhanced.mp3", "_base.mp3", ".wav"]:
            audio_path = os.path.join(TTS_OUTPUT_DIR, f"{session_id}{audio_suffix}")
            if os.path.exists(audio_path):
                assets["local_paths"]["audio"] = audio_path
                assets["status"]["audio_exists"] = True
                break
        else:
            assets["status"]["audio_exists"] = False

        return assets

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get assets: {str(e)}")

@app.get("/api/assets/batch")
async def get_batch_assets(session_ids: str):
    """Get assets for multiple sessions (comma-separated session IDs)"""

    session_id_list = [sid.strip() for sid in session_ids.split(',')]
    batch_assets = {}

    for session_id in session_id_list:
        try:
            assets = await get_session_assets(session_id)
            batch_assets[session_id] = assets
        except HTTPException:
            batch_assets[session_id] = {"error": "Session not found"}
        except Exception as e:
            batch_assets[session_id] = {"error": str(e)}

    return {
        "total_sessions": len(session_id_list),
        "successful": len([a for a in batch_assets.values() if "error" not in a]),
        "failed": len([a for a in batch_assets.values() if "error" in a]),
        "assets": batch_assets
    }

@app.get("/api/assets/compressed/{session_id}")
async def get_compressed_assets(session_id: str, format: str = "mp3"):
    """Get compressed audio asset for a session"""

    # Look for compressed audio files
    compressed_paths = [
        os.path.join(TTS_OUTPUT_DIR, f"{session_id}_optimized.mp3"),
        os.path.join(TTS_OUTPUT_DIR, f"{session_id}_enhanced.mp3"),
        os.path.join(TTS_OUTPUT_DIR, f"{session_id}_base.mp3")
    ]

    for path in compressed_paths:
        if os.path.exists(path):
            return FileResponse(
                path=path,
                filename=f"compressed_audio_{session_id}.mp3",
                media_type="audio/mpeg"
            )

    raise HTTPException(status_code=404, detail="Compressed audio not found")

@app.get("/api/lessons/assets/mapping")
async def get_lesson_asset_mapping():
    """Get mapping of all lessons to their assets"""

    lessons_index = lesson_manager.get_lessons_index()
    asset_mapping = {
        "total_lessons": lessons_index["total_lessons"],
        "lessons_with_assets": 0,
        "lessons_without_assets": 0,
        "mapping": {}
    }

    for lesson_id, lesson_info in lessons_index["lessons"].items():
        lesson_data = lesson_manager.get_lesson(lesson_id)

        if lesson_data:
            has_assets = bool(lesson_data["assets"]["audio_url"])

            asset_mapping["mapping"][lesson_id] = {
                "title": lesson_data["title"],
                "category": lesson_data["category"],
                "status": lesson_data["status"],
                "has_assets": has_assets,
                "assets": lesson_data["assets"],
                "cloud_urls": {
                    "audio_url": lesson_data["assets"]["audio_url"],
                    "video_url": lesson_data["assets"]["video_url"],
                    "metadata_url": lesson_data["assets"]["metadata_url"]
                }
            }

            if has_assets:
                asset_mapping["lessons_with_assets"] += 1
            else:
                asset_mapping["lessons_without_assets"] += 1

    return asset_mapping

@app.get("/api/assets/download/{session_id}/{asset_type}")
async def download_asset(session_id: str, asset_type: str):
    """Download specific asset type (audio, video, metadata, sync_map)"""

    if asset_type == "video":
        file_path = os.path.join(RESULTS_DIR, f"{session_id}.mp4")
        media_type = "video/mp4"
        filename = f"video_{session_id}.mp4"

    elif asset_type == "audio":
        # Find the best audio file
        for suffix in ["_optimized.mp3", "_enhanced.mp3", "_base.mp3"]:
            file_path = os.path.join(TTS_OUTPUT_DIR, f"{session_id}{suffix}")
            if os.path.exists(file_path):
                break
        else:
            raise HTTPException(status_code=404, detail="Audio file not found")

        media_type = "audio/mpeg"
        filename = f"audio_{session_id}.mp3"

    elif asset_type == "metadata":
        file_path = os.path.join(RESULTS_DIR, f"metadata_{session_id}.json")
        media_type = "application/json"
        filename = f"metadata_{session_id}.json"

    elif asset_type == "sync_map":
        file_path = f"sync_maps/sync_map_{session_id}.json"
        media_type = "application/json"
        filename = f"sync_map_{session_id}.json"

    else:
        raise HTTPException(status_code=400, detail="Invalid asset type. Use: audio, video, metadata, sync_map")

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"{asset_type.title()} file not found")

    return FileResponse(
        path=file_path,
        filename=filename,
        media_type=media_type
    )

# Utility function to create default transition tones
def create_default_transition_tones():
    """Create default transition tone files if they don't exist"""

    if not os.path.exists(TRANSITION_SOUNDS_DIR):
        os.makedirs(TRANSITION_SOUNDS_DIR, exist_ok=True)

    # Generate and save default tones for each emotion
    for emotion, params in tts_config.transition_tones.items():
        tone_path = os.path.join(TRANSITION_SOUNDS_DIR, f"{emotion}_tone.wav")

        if not os.path.exists(tone_path):
            try:
                # Generate synthetic tone
                tone = enhanced_tts._generate_synthetic_tone(params)
                tone.export(tone_path, format="wav")
                print(f"[OK] Created transition tone: {emotion}_tone.wav")
            except Exception as e:
                print(f"[WARNING] Failed to create transition tone {emotion}: {e}")

# Initialize default transition tones on startup
create_default_transition_tones()

if __name__ == "__main__":
    import uvicorn
    print("🎭 Enhanced Avatar Engine with Emotional TTS")
    print("=" * 50)
    print(f"   🎵 Audio compression: {'✅' if tts_config.enable_audio_compression else '❌'}")
    print(f"   🔔 Transition tones: {'✅' if tts_config.enable_transition_tones else '❌'}")
    print(f"   📊 Compression quality: {tts_config.compression_quality} kbps")
    print(f"   🎚️  Target sample rate: {tts_config.target_sample_rate} Hz")
    print(f"   😊 Available emotions: {len(tts_config.transition_tones)}")
    print(f"   🧠 Multimodal sentiment: {'✅' if SENTIMENT_INTEGRATION_AVAILABLE else '❌'}")
    print(f"   🎯 LoRA TTS Engine: {'✅' if LORA_TTS_AVAILABLE else '⚠️  (using fallback)'}")
    print(f"   🔄 Emotional Fallback: {'✅' if EMOTIONAL_FALLBACK_AVAILABLE else '❌'}")
    print(f"   🛡️  Fallback pipeline: ✅ (Enhanced gTTS with emotional processing)")
    print("\n🚀 TTS Pipeline Priority:")
    print("   1. LoRA TTS with emotional control")
    print("   2. Enhanced gTTS with emotional processing")
    print("   3. Basic gTTS with optimizations")
    print(f"\n🌐 Server starting at: http://192.168.1.102:8002")
    print("=" * 50)
    uvicorn.run(app, host="192.168.1.102", port=8002)
 