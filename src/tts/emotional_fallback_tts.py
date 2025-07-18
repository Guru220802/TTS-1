"""
Emotional Fallback TTS System
Advanced emotional processing for gTTS when LoRA is not available
"""

import os
import numpy as np
import logging
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import librosa
import soundfile as sf
from pydub import AudioSegment
from pydub.effects import normalize, compress_dynamic_range
import scipy.signal
from scipy import interpolate
import uuid


@dataclass
class EmotionalProcessingConfig:
    """Configuration for emotional audio processing"""
    # Pitch modification
    pitch_shift_semitones: Dict[str, float] = None
    
    # Speed/tempo modification
    speed_multiplier: Dict[str, float] = None
    
    # Formant shifting (voice character)
    formant_shift: Dict[str, float] = None
    
    # Dynamic range and energy
    energy_boost: Dict[str, float] = None
    compression_ratio: Dict[str, float] = None
    
    # Spectral filtering
    brightness: Dict[str, float] = None  # High frequency emphasis
    warmth: Dict[str, float] = None      # Low frequency emphasis
    
    # Prosodic modifications
    pause_length_multiplier: Dict[str, float] = None
    emphasis_strength: Dict[str, float] = None


class EmotionalAudioProcessor:
    """Advanced audio processor for emotional TTS enhancement"""
    
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
        self.logger = logging.getLogger(__name__)
        
        # Initialize emotional processing configurations
        self.config = self._initialize_emotional_configs()
    
    def _initialize_emotional_configs(self) -> EmotionalProcessingConfig:
        """Initialize emotional processing parameters"""
        return EmotionalProcessingConfig(
            pitch_shift_semitones={
                'joyful': 2.5,
                'peaceful': -1.0,
                'balanced': 0.0,
                'enthusiastic': 3.5,
                'contemplative': -0.5,
                'warm': 0.5,
                'inspiring': 2.0,
                'confident': 1.0,
                'grounded': -1.5,
                'soothing': -2.0
            },
            speed_multiplier={
                'joyful': 1.15,
                'peaceful': 0.85,
                'balanced': 1.0,
                'enthusiastic': 1.25,
                'contemplative': 0.8,
                'warm': 0.95,
                'inspiring': 1.1,
                'confident': 1.05,
                'grounded': 0.9,
                'soothing': 0.8
            },
            formant_shift={
                'joyful': 1.1,
                'peaceful': 0.95,
                'balanced': 1.0,
                'enthusiastic': 1.15,
                'contemplative': 0.98,
                'warm': 1.05,
                'inspiring': 1.08,
                'confident': 1.02,
                'grounded': 0.92,
                'soothing': 0.9
            },
            energy_boost={
                'joyful': 1.3,
                'peaceful': 0.8,
                'balanced': 1.0,
                'enthusiastic': 1.4,
                'contemplative': 0.9,
                'warm': 1.1,
                'inspiring': 1.2,
                'confident': 1.15,
                'grounded': 0.95,
                'soothing': 0.75
            },
            compression_ratio={
                'joyful': 0.7,
                'peaceful': 0.3,
                'balanced': 0.5,
                'enthusiastic': 0.8,
                'contemplative': 0.4,
                'warm': 0.5,
                'inspiring': 0.6,
                'confident': 0.6,
                'grounded': 0.4,
                'soothing': 0.2
            },
            brightness={
                'joyful': 1.3,
                'peaceful': 0.8,
                'balanced': 1.0,
                'enthusiastic': 1.4,
                'contemplative': 0.9,
                'warm': 0.9,
                'inspiring': 1.2,
                'confident': 1.1,
                'grounded': 0.85,
                'soothing': 0.7
            },
            warmth={
                'joyful': 1.1,
                'peaceful': 1.3,
                'balanced': 1.0,
                'enthusiastic': 1.0,
                'contemplative': 1.2,
                'warm': 1.4,
                'inspiring': 1.1,
                'confident': 1.0,
                'grounded': 1.2,
                'soothing': 1.5
            },
            pause_length_multiplier={
                'joyful': 0.8,
                'peaceful': 1.3,
                'balanced': 1.0,
                'enthusiastic': 0.7,
                'contemplative': 1.4,
                'warm': 1.1,
                'inspiring': 0.9,
                'confident': 0.9,
                'grounded': 1.2,
                'soothing': 1.5
            },
            emphasis_strength={
                'joyful': 1.2,
                'peaceful': 0.7,
                'balanced': 1.0,
                'enthusiastic': 1.4,
                'contemplative': 0.8,
                'warm': 0.9,
                'inspiring': 1.3,
                'confident': 1.1,
                'grounded': 0.9,
                'soothing': 0.6
            }
        )
    
    def process_emotional_audio(self, audio_path: str, emotion: str, output_path: str) -> str:
        """Apply comprehensive emotional processing to audio"""
        try:
            self.logger.info(f"Processing audio for emotion: {emotion}")
            
            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Apply emotional transformations
            processed_audio = self._apply_emotional_transformations(audio, emotion)
            
            # Save processed audio
            sf.write(output_path, processed_audio, self.sample_rate)
            
            self.logger.info(f"✅ Emotional processing complete: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Emotional processing failed: {e}")
            raise
    
    def _apply_emotional_transformations(self, audio: np.ndarray, emotion: str) -> np.ndarray:
        """Apply all emotional transformations to audio"""
        
        # Get emotion parameters
        emotion = emotion.lower()
        if emotion not in self.config.pitch_shift_semitones:
            emotion = 'balanced'
        
        processed = audio.copy()
        
        # 1. Pitch shifting
        pitch_shift = self.config.pitch_shift_semitones[emotion]
        if abs(pitch_shift) > 0.1:
            processed = self._pitch_shift(processed, pitch_shift)
        
        # 2. Speed/tempo modification
        speed_mult = self.config.speed_multiplier[emotion]
        if abs(speed_mult - 1.0) > 0.05:
            processed = self._time_stretch(processed, speed_mult)
        
        # 3. Formant shifting (voice character)
        formant_shift = self.config.formant_shift[emotion]
        if abs(formant_shift - 1.0) > 0.02:
            processed = self._formant_shift(processed, formant_shift)
        
        # 4. Spectral filtering (brightness/warmth)
        brightness = self.config.brightness[emotion]
        warmth = self.config.warmth[emotion]
        processed = self._spectral_filtering(processed, brightness, warmth)
        
        # 5. Dynamic range processing
        energy_boost = self.config.energy_boost[emotion]
        compression = self.config.compression_ratio[emotion]
        processed = self._dynamic_processing(processed, energy_boost, compression)
        
        # 6. Prosodic modifications (pause enhancement)
        pause_mult = self.config.pause_length_multiplier[emotion]
        if abs(pause_mult - 1.0) > 0.1:
            processed = self._enhance_pauses(processed, pause_mult)
        
        return processed
    
    def _pitch_shift(self, audio: np.ndarray, semitones: float) -> np.ndarray:
        """Shift pitch by semitones"""
        try:
            return librosa.effects.pitch_shift(
                audio, 
                sr=self.sample_rate, 
                n_steps=semitones,
                bins_per_octave=12
            )
        except Exception as e:
            self.logger.warning(f"Pitch shift failed: {e}")
            return audio
    
    def _time_stretch(self, audio: np.ndarray, rate: float) -> np.ndarray:
        """Stretch or compress time without changing pitch"""
        try:
            return librosa.effects.time_stretch(audio, rate=rate)
        except Exception as e:
            self.logger.warning(f"Time stretch failed: {e}")
            return audio
    
    def _formant_shift(self, audio: np.ndarray, shift_factor: float) -> np.ndarray:
        """Shift formants to change voice character"""
        try:
            # Simple formant shifting using spectral envelope manipulation
            stft = librosa.stft(audio, n_fft=2048, hop_length=512)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Shift spectral envelope
            freq_bins = magnitude.shape[0]
            shifted_magnitude = np.zeros_like(magnitude)
            
            for i in range(freq_bins):
                source_bin = int(i / shift_factor)
                if 0 <= source_bin < freq_bins:
                    shifted_magnitude[i] = magnitude[source_bin]
            
            # Reconstruct audio
            shifted_stft = shifted_magnitude * np.exp(1j * phase)
            return librosa.istft(shifted_stft, hop_length=512)
            
        except Exception as e:
            self.logger.warning(f"Formant shift failed: {e}")
            return audio
    
    def _spectral_filtering(self, audio: np.ndarray, brightness: float, warmth: float) -> np.ndarray:
        """Apply spectral filtering for brightness and warmth"""
        try:
            # Design filters
            nyquist = self.sample_rate // 2
            
            # High-frequency emphasis for brightness
            if abs(brightness - 1.0) > 0.05:
                high_freq = 3000  # Hz
                b_high, a_high = scipy.signal.butter(2, high_freq / nyquist, btype='high')
                high_emphasis = scipy.signal.filtfilt(b_high, a_high, audio)
                audio = audio + (brightness - 1.0) * 0.3 * high_emphasis
            
            # Low-frequency emphasis for warmth
            if abs(warmth - 1.0) > 0.05:
                low_freq = 800  # Hz
                b_low, a_low = scipy.signal.butter(2, low_freq / nyquist, btype='low')
                low_emphasis = scipy.signal.filtfilt(b_low, a_low, audio)
                audio = audio + (warmth - 1.0) * 0.2 * low_emphasis
            
            return audio
            
        except Exception as e:
            self.logger.warning(f"Spectral filtering failed: {e}")
            return audio
    
    def _dynamic_processing(self, audio: np.ndarray, energy_boost: float, compression: float) -> np.ndarray:
        """Apply dynamic range processing"""
        try:
            # Energy boost
            audio = audio * energy_boost
            
            # Compression (simple)
            if compression > 0.1:
                threshold = 0.1
                ratio = 1.0 / (1.0 - compression)
                
                # Find peaks above threshold
                peaks = np.abs(audio) > threshold
                
                # Apply compression to peaks
                audio[peaks] = np.sign(audio[peaks]) * (
                    threshold + (np.abs(audio[peaks]) - threshold) / ratio
                )
            
            # Normalize to prevent clipping
            max_val = np.max(np.abs(audio))
            if max_val > 0.95:
                audio = audio * (0.95 / max_val)
            
            return audio
            
        except Exception as e:
            self.logger.warning(f"Dynamic processing failed: {e}")
            return audio
    
    def _enhance_pauses(self, audio: np.ndarray, pause_multiplier: float) -> np.ndarray:
        """Enhance pauses between words/phrases"""
        try:
            # Detect silence regions
            silence_threshold = 0.01
            frame_length = 1024
            hop_length = 512
            
            # Calculate RMS energy
            rms = librosa.feature.rms(
                y=audio, 
                frame_length=frame_length, 
                hop_length=hop_length
            )[0]
            
            # Find silent frames
            silent_frames = rms < silence_threshold
            
            # Convert frame indices to sample indices
            frame_times = librosa.frames_to_samples(
                np.arange(len(rms)), 
                hop_length=hop_length
            )
            
            # Extend silent regions
            if pause_multiplier != 1.0:
                extended_audio = []
                last_sample = 0
                
                for i, is_silent in enumerate(silent_frames):
                    current_sample = frame_times[i] if i < len(frame_times) else len(audio)
                    
                    # Add non-silent audio
                    extended_audio.append(audio[last_sample:current_sample])
                    
                    # Extend silent regions
                    if is_silent and i < len(frame_times) - 1:
                        silence_length = frame_times[i + 1] - current_sample
                        extended_silence_length = int(silence_length * pause_multiplier)
                        silence = np.zeros(extended_silence_length)
                        extended_audio.append(silence)
                    
                    last_sample = current_sample
                
                # Add remaining audio
                if last_sample < len(audio):
                    extended_audio.append(audio[last_sample:])
                
                return np.concatenate(extended_audio)
            
            return audio
            
        except Exception as e:
            self.logger.warning(f"Pause enhancement failed: {e}")
            return audio


class EmotionalFallbackTTS:
    """Enhanced gTTS with emotional processing fallback"""
    
    def __init__(self, sample_rate: int = 22050):
        self.processor = EmotionalAudioProcessor(sample_rate)
        self.logger = logging.getLogger(__name__)
    
    def generate_emotional_speech(self, text: str, emotion: str, language: str = 'en') -> str:
        """Generate emotionally enhanced speech using gTTS + processing"""
        session_id = str(uuid.uuid4())
        
        try:
            # Generate base audio with gTTS
            from gtts import gTTS
            
            base_path = f"tts/tts_outputs/{session_id}_base.mp3"
            os.makedirs(os.path.dirname(base_path), exist_ok=True)
            
            tts = gTTS(text=text, lang=language, slow=False)
            tts.save(base_path)
            
            # Convert to WAV for processing
            wav_path = f"tts/tts_outputs/{session_id}_base.wav"
            audio = AudioSegment.from_mp3(base_path)
            audio.export(wav_path, format="wav")
            
            # Apply emotional processing
            processed_path = f"tts/tts_outputs/{session_id}_emotional.wav"
            self.processor.process_emotional_audio(wav_path, emotion, processed_path)
            
            # Convert back to MP3
            final_path = f"tts/tts_outputs/{session_id}_emotional.mp3"
            processed_audio = AudioSegment.from_wav(processed_path)
            processed_audio.export(final_path, format="mp3", bitrate="128k")
            
            self.logger.info(f"✅ Emotional fallback TTS generated: {final_path}")
            return final_path
            
        except Exception as e:
            self.logger.error(f"Emotional fallback TTS failed: {e}")
            raise


# Global instance
_emotional_fallback_tts = None

def get_emotional_fallback_tts() -> EmotionalFallbackTTS:
    """Get global emotional fallback TTS instance"""
    global _emotional_fallback_tts
    if _emotional_fallback_tts is None:
        _emotional_fallback_tts = EmotionalFallbackTTS()
    return _emotional_fallback_tts
