"""
Sync Map Generator for TTS Integration
Creates precise timestamp data for UI controls and visual synchronization
"""

import json
import os
import librosa
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import logging
from datetime import datetime, timezone
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SyncMapGenerator:
    """Generates sync maps and timestamp data for audio-visual synchronization"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def generate_sync_map(self, audio_path: str, text: str, session_id: str) -> Dict[str, Any]:
        """Generate comprehensive sync map for audio file"""
        
        try:
            # Load audio file
            y, sr = librosa.load(audio_path)
            duration = librosa.get_duration(y=y, sr=sr)
            
            # Generate word-level timestamps
            word_timestamps = self._generate_word_timestamps(text, duration)
            
            # Generate sentence-level timestamps
            sentence_timestamps = self._generate_sentence_timestamps(text, duration)
            
            # Generate audio features for sync
            audio_features = self._extract_audio_features(y, sr)
            
            # Generate frame-level data for video sync
            frame_data = self._generate_frame_data(y, sr, duration)
            
            sync_map = {
                "session_id": session_id,
                "audio_path": audio_path,
                "text": text,
                "duration_seconds": duration,
                "sample_rate": sr,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "word_timestamps": word_timestamps,
                "sentence_timestamps": sentence_timestamps,
                "audio_features": audio_features,
                "frame_data": frame_data,
                "ui_controls": self._generate_ui_controls(word_timestamps, sentence_timestamps),
                "visual_sync": self._generate_visual_sync_data(frame_data, duration)
            }
            
            self.logger.info(f"✅ Generated sync map for {session_id}")
            return sync_map
            
        except Exception as e:
            self.logger.error(f"❌ Sync map generation failed: {e}")
            return self._generate_fallback_sync_map(text, session_id)
    
    def _generate_word_timestamps(self, text: str, duration: float) -> List[Dict[str, Any]]:
        """Generate word-level timestamps"""
        words = text.split()
        word_count = len(words)
        
        if word_count == 0:
            return []
        
        # Estimate timing based on average speaking rate (150 words per minute)
        words_per_second = 2.5
        estimated_duration = word_count / words_per_second
        
        # Scale to actual duration
        time_scale = duration / max(estimated_duration, 1.0)
        
        word_timestamps = []
        current_time = 0.0
        
        for i, word in enumerate(words):
            # Estimate word duration based on length and complexity
            base_duration = (len(word) + 2) / 10.0  # Base timing
            word_duration = base_duration * time_scale
            
            # Add pauses for punctuation
            if word.endswith(('.', '!', '?')):
                word_duration += 0.3 * time_scale
            elif word.endswith(','):
                word_duration += 0.15 * time_scale
            
            word_data = {
                "word": word.strip('.,!?;:'),
                "start_time": round(current_time, 3),
                "end_time": round(current_time + word_duration, 3),
                "duration": round(word_duration, 3),
                "index": i,
                "confidence": 0.85  # Estimated confidence
            }
            
            word_timestamps.append(word_data)
            current_time += word_duration
        
        return word_timestamps
    
    def _generate_sentence_timestamps(self, text: str, duration: float) -> List[Dict[str, Any]]:
        """Generate sentence-level timestamps"""
        # Split text into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return []
        
        sentence_timestamps = []
        current_time = 0.0
        
        for i, sentence in enumerate(sentences):
            # Estimate sentence duration based on word count
            word_count = len(sentence.split())
            sentence_duration = (word_count / 2.5) * (duration / len(text.split()))
            
            sentence_data = {
                "sentence": sentence,
                "start_time": round(current_time, 3),
                "end_time": round(current_time + sentence_duration, 3),
                "duration": round(sentence_duration, 3),
                "index": i,
                "word_count": word_count
            }
            
            sentence_timestamps.append(sentence_data)
            current_time += sentence_duration
        
        return sentence_timestamps
    
    def _extract_audio_features(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Extract audio features for synchronization"""
        
        # Extract basic features
        rms = librosa.feature.rms(y=y)[0]
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
        
        # Tempo and beat tracking
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        
        # Onset detection
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
        
        return {
            "tempo": float(tempo),
            "beats": beats.tolist(),
            "onset_times": onset_times.tolist(),
            "rms_energy": {
                "mean": float(np.mean(rms)),
                "std": float(np.std(rms)),
                "max": float(np.max(rms)),
                "values": rms.tolist()
            },
            "spectral_centroid": {
                "mean": float(np.mean(spectral_centroids)),
                "values": spectral_centroids.tolist()
            },
            "zero_crossing_rate": {
                "mean": float(np.mean(zero_crossing_rate)),
                "values": zero_crossing_rate.tolist()
            }
        }
    
    def _generate_frame_data(self, y: np.ndarray, sr: int, duration: float) -> Dict[str, Any]:
        """Generate frame-level data for video synchronization"""
        
        # Standard video frame rate
        fps = 25
        total_frames = int(duration * fps)
        
        # Generate frame timestamps
        frame_times = [i / fps for i in range(total_frames)]
        
        # Map audio features to video frames
        hop_length = 512
        frame_length = 2048
        
        # Extract frame-level features
        stft = librosa.stft(y, hop_length=hop_length, n_fft=frame_length)
        magnitude = np.abs(stft)
        
        # Resample to video frame rate
        audio_frames = magnitude.shape[1]
        video_frame_indices = np.linspace(0, audio_frames-1, total_frames, dtype=int)
        
        frame_data = {
            "fps": fps,
            "total_frames": total_frames,
            "frame_times": frame_times,
            "audio_sync": {
                "hop_length": hop_length,
                "frame_length": frame_length,
                "magnitude_per_frame": magnitude[:, video_frame_indices].mean(axis=0).tolist()
            }
        }
        
        return frame_data
    
    def _generate_ui_controls(self, word_timestamps: List[Dict], sentence_timestamps: List[Dict]) -> Dict[str, Any]:
        """Generate UI control data for Rishabh's frontend"""
        
        # Create progress markers
        progress_markers = []
        for sentence in sentence_timestamps:
            progress_markers.append({
                "time": sentence["start_time"],
                "progress_percent": (sentence["start_time"] / sentence_timestamps[-1]["end_time"]) * 100,
                "label": sentence["sentence"][:50] + "..." if len(sentence["sentence"]) > 50 else sentence["sentence"]
            })
        
        # Create word highlighting data
        word_highlights = []
        for word in word_timestamps:
            word_highlights.append({
                "start_time": word["start_time"],
                "end_time": word["end_time"],
                "word": word["word"],
                "index": word["index"]
            })
        
        return {
            "progress_markers": progress_markers,
            "word_highlights": word_highlights,
            "total_duration": sentence_timestamps[-1]["end_time"] if sentence_timestamps else 0,
            "playback_controls": {
                "skip_forward": 10,  # seconds
                "skip_backward": 10,  # seconds
                "speed_options": [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
            }
        }
    
    def _generate_visual_sync_data(self, frame_data: Dict, duration: float) -> Dict[str, Any]:
        """Generate visual synchronization data for Shashank's animations"""
        
        return {
            "animation_keyframes": [
                {"time": 0.0, "type": "start", "intensity": 0.5},
                {"time": duration * 0.25, "type": "emphasis", "intensity": 0.8},
                {"time": duration * 0.5, "type": "peak", "intensity": 1.0},
                {"time": duration * 0.75, "type": "emphasis", "intensity": 0.8},
                {"time": duration, "type": "end", "intensity": 0.3}
            ],
            "lip_sync_frames": frame_data["frame_times"],
            "emotion_transitions": [
                {"time": 0.0, "emotion": "neutral"},
                {"time": duration * 0.3, "emotion": "engaged"},
                {"time": duration * 0.7, "emotion": "expressive"},
                {"time": duration, "emotion": "neutral"}
            ]
        }
    
    def _generate_fallback_sync_map(self, text: str, session_id: str) -> Dict[str, Any]:
        """Generate basic sync map when audio analysis fails"""
        
        words = text.split()
        estimated_duration = len(words) / 2.5  # 150 words per minute
        
        return {
            "session_id": session_id,
            "text": text,
            "duration_seconds": estimated_duration,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "fallback": True,
            "word_timestamps": self._generate_word_timestamps(text, estimated_duration),
            "sentence_timestamps": self._generate_sentence_timestamps(text, estimated_duration),
            "ui_controls": {
                "progress_markers": [{"time": 0, "progress_percent": 0, "label": "Start"}],
                "word_highlights": [],
                "total_duration": estimated_duration
            }
        }
    
    def save_sync_map(self, sync_map: Dict[str, Any], output_dir: str = "sync_maps") -> str:
        """Save sync map to file"""
        
        os.makedirs(output_dir, exist_ok=True)
        session_id = sync_map["session_id"]
        output_path = os.path.join(output_dir, f"sync_map_{session_id}.json")
        
        with open(output_path, 'w') as f:
            json.dump(sync_map, f, indent=2)
        
        self.logger.info(f"✅ Saved sync map to {output_path}")
        return output_path

# Global sync map generator instance
sync_map_generator = SyncMapGenerator()
