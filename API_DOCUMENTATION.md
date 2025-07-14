# TTS-LipSync-Translation API Documentation

## Overview

This API provides comprehensive Text-to-Speech (TTS) services with multilingual support, avatar-based lip-sync video generation, and translation capabilities. The system consists of two main services:

1. **Basic TTS Service** (`tts.py`) - Simple text-to-speech audio generation
2. **Advanced Avatar Engine** (`avatar_engine.py`) - Full TTS + Translation + Lip-sync video generation

## Base URLs

- **Basic TTS Service**: `http://192.168.0.119:8001`
- **Avatar Engine Service**: `http://192.168.0.125:8001`

---

## Basic TTS Service Endpoints

### 1. Health Check
**GET** `/`

Returns service status.

**Response:**
```json
{
  "message": "TTS Service is running"
}
```

### 2. Generate Audio
**POST** `/api/generate`

Converts text to speech and returns audio file information.

**Request:**
- **Content-Type**: `application/x-www-form-urlencoded`
- **Body Parameters**:
  - `text` (string, required): Text to convert to speech (max recommended: 500 characters)

**Response:**
```json
{
  "status": "success",
  "audio_url": "/api/audio/{filename}",
  "filename": "uuid.mp3"
}
```

**Error Responses:**
- `400`: Text is required
- `500`: Audio generation failed

### 3. Retrieve Audio File
**GET** `/api/audio/{filename}`

Downloads the generated audio file.

**Parameters:**
- `filename` (string): The filename returned from `/api/generate`

**Response:**
- **Content-Type**: `audio/mpeg`
- **Body**: MP3 audio file

**Error Responses:**
- `404`: Audio file not found

### 4. List Audio Files
**GET** `/api/list-audio-files`

Returns list of all generated audio files.

**Response:**
```json
{
  "audio_files": ["file1.mp3", "file2.mp3"],
  "count": 2
}
```

---

## Avatar Engine Service Endpoints

### 1. Health Check
**GET** `/`

Returns service status.

**Response:**
```json
{
  "message": "TTS-LipSync-Translation API running"
}
```

### 2. Generate Lip-Sync Video
**POST** `/api/generate-and-sync`

Complete pipeline: Text → Translation → TTS → Gender Detection → Avatar Selection → Lip-sync Video

**Request:**
- **Content-Type**: `application/x-www-form-urlencoded`
- **Body Parameters**:
  - `text` (string, required): Text to convert (max 500 characters, auto-truncated)
  - `target_lang` (string, optional, default: 'en'): Target language code

**Supported Languages:**
- `en` - English
- `hi` - Hindi (Devanagari)
- `mr` - Marathi (Devanagari)
- `ta` - Tamil (Tamil script)
- `te` - Telugu (Telugu script)
- `kn` - Kannada (Kannada script)
- `ml` - Malayalam (Malayalam script)
- `gu` - Gujarati (Gujarati script)
- `bn` - Bengali (Bengali script)
- `pa` - Punjabi (Gurmukhi script)
- `es` - Spanish
- `fr` - French
- `de` - German
- `zh` - Chinese
- `ja` - Japanese
- `ru` - Russian
- `ar` - Arabic
- `pt` - Portuguese
- `it` - Italian

**Response:**
- **Content-Type**: `video/mp4`
- **Body**: MP4 video file with lip-synced avatar
- **Filename**: `lipsync_{session_id}.mp4`

**Process Flow:**
1. Text validation and truncation (500 char limit)
2. Translation (if target_lang != 'en')
3. TTS generation using gTTS
4. Audio format conversion (MP3 → WAV)
5. Gender detection from voice
6. Avatar selection based on gender
7. Wav2Lip processing for lip-sync
8. Metadata generation and storage

**Error Responses:**
- `400`: Text is required
- `500`: Translation failed / FFmpeg/Wav2Lip failed / Unexpected error

### 3. Get Video Metadata
**GET** `/api/metadata/{session_id}`

Retrieves metadata for a generated video.

**Parameters:**
- `session_id` (string): UUID of the video session

**Response:**
```json
{
  "session_id": "uuid",
  "language": "hi",
  "language_name": "Hindi",
  "script": "Devanagari",
  "text_length": 45,
  "gender": "female",
  "avatar": "pht1.jpg",
  "timestamp": "2024-01-01T12:00:00Z",
  "video_format": "mp4"
}
```

**Error Responses:**
- `404`: Metadata not found
- `500`: Failed to load metadata

---

## Technical Implementation Details

### Gender Detection
- Uses trained Keras model (`gender-recognition-by-voice/results/model.h5`)
- Extracts audio features using librosa
- Predicts gender from voice characteristics
- Fallback to "default" if model unavailable

### Avatar Selection
- **Female avatars**: `pht1.jpg`, `pht2.jpg`
- **Male avatars**: `pht3.jpg`, `pht4.jpg`
- **Default**: `pht1.jpg`
- Random selection within gender category

### Translation Service
- Uses Google Gemini API for translation
- Confidence scoring for translation quality
- Script validation for non-Latin languages
- Retry logic with exponential backoff

### File Storage
- **TTS outputs**: `tts/tts_outputs/`
- **Video results**: `results/`
- **Metadata**: `results/metadata_{session_id}.json`
- **Avatars**: `avatars/`

### Dependencies
- FastAPI for API framework
- gTTS for text-to-speech
- Wav2Lip for lip-sync generation
- Keras/TensorFlow for gender detection
- librosa for audio processing
- Google Gemini for translation

---

## Error Handling

All endpoints return structured error responses:

```json
{
  "detail": "Error description"
}
```

Common HTTP status codes:
- `200`: Success
- `400`: Bad Request (missing/invalid parameters)
- `404`: Not Found (file/resource not found)
- `500`: Internal Server Error (processing failures)

---

## Rate Limiting & Performance

- Text input limited to 500 characters
- Translation includes retry logic (max 3 attempts)
- Video generation timeout: ~300 seconds
- Concurrent request handling via FastAPI async

---

## Security Considerations

- Input validation and sanitization
- File path validation
- Error message sanitization
- No authentication currently implemented (consider adding for production)
