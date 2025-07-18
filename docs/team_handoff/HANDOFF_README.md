# TTS-LipSync-Translation System Handoff Documentation

## Project Overview

This document outlines the handoff of the TTS (Text-to-Speech) system with multilingual translation and avatar lip-sync capabilities. The system is now ready for integration and UI enhancement by the designated team members.

## System Architecture

```
User Input (Text) → Translation → TTS → Gender Detection → Avatar Selection → Lip-Sync → Video Output
```

### Current Implementation Status ✅

- ✅ **Core TTS Engine**: Functional with pyttsx3 and gTTS
- ✅ **Translation Service**: Google Gemini API integration with 19 languages
- ✅ **Gender Detection**: Voice-based gender classification
- ✅ **Avatar System**: Gender-specific avatar selection
- ✅ **Lip-Sync Generation**: Wav2Lip integration
- ✅ **API Endpoints**: RESTful API with FastAPI
- ✅ **Metadata System**: Video generation tracking
- ✅ **Error Handling**: Comprehensive error responses

---

## Team Member Responsibilities

### 🔧 Vedant - API Integration Layer

**Primary Focus**: Backend API integration and service orchestration

#### Current State
- Two separate services running:
  - Basic TTS: `http://192.168.0.119:8001`
  - Avatar Engine: `http://192.168.0.125:8001`

#### Your Tasks
1. **Service Consolidation**
   - Merge both services into unified API
   - Implement service discovery/load balancing if needed
   - Add authentication/authorization layer

2. **API Enhancement**
   - Add batch processing capabilities
   - Implement request queuing for high load
   - Add API versioning (`/v1/`, `/v2/`)
   - Create webhook support for async processing

3. **Database Integration**
   - Replace file-based metadata with database
   - Implement user session management
   - Add request logging and analytics

4. **Performance Optimization**
   - Add caching layer (Redis recommended)
   - Implement connection pooling
   - Add request rate limiting

#### Key Files to Work With
- `avatar_engine.py` - Main service logic
- `tts.py` - Basic TTS service
- `translation_agent.py` - Translation logic

#### Integration Points
```python
# Example API client for your integration
import requests

class TTSClient:
    def __init__(self, base_url):
        self.base_url = base_url
    
    def generate_video(self, text, language='en'):
        response = requests.post(
            f"{self.base_url}/api/generate-and-sync",
            data={"text": text, "target_lang": language}
        )
        return response
```

---

### 🎨 Rishabh - Frontend Hooks & UI Event Triggers

**Primary Focus**: Frontend integration and user interaction handling

#### Current State
- Streamlit demo app in `avatar.py`
- Basic form-based interface

#### Your Tasks
1. **React/Vue Component Development**
   - Create reusable TTS components
   - Implement real-time progress indicators
   - Add language selection dropdown
   - Build audio/video player components

2. **Event Handling System**
   - Text input validation and character counting
   - Real-time language detection
   - Progress tracking for video generation
   - Error state management

3. **State Management**
   - Implement Redux/Vuex for TTS state
   - Handle async operations (video generation)
   - Manage user preferences (language, voice)
   - Cache generated content

4. **WebSocket Integration**
   - Real-time progress updates
   - Live status notifications
   - Queue position tracking

#### Frontend Hook Examples
```javascript
// React Hook Example
import { useState, useCallback } from 'react';

export const useTTS = () => {
  const [isGenerating, setIsGenerating] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState(null);

  const generateVideo = useCallback(async (text, language) => {
    setIsGenerating(true);
    setError(null);
    
    try {
      const response = await fetch('/api/generate-and-sync', {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: new URLSearchParams({ text, target_lang: language })
      });
      
      if (!response.ok) throw new Error('Generation failed');
      
      const videoBlob = await response.blob();
      return URL.createObjectURL(videoBlob);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsGenerating(false);
    }
  }, []);

  return { generateVideo, isGenerating, progress, error };
};
```

#### UI Components to Build
- `<TTSInput />` - Text input with validation
- `<LanguageSelector />` - Language dropdown
- `<ProgressIndicator />` - Generation progress
- `<VideoPlayer />` - Generated video display
- `<ErrorBoundary />` - Error handling

---

### 🎯 Shashank - UX Refinement & UI Playback Controls

**Primary Focus**: User experience optimization and media controls

#### Current State
- Basic video download functionality
- Simple text input interface

#### Your Tasks
1. **Advanced Media Controls**
   - Custom video player with scrubbing
   - Playback speed controls (0.5x, 1x, 1.5x, 2x)
   - Volume controls and mute functionality
   - Fullscreen mode support

2. **UX Enhancements**
   - Drag-and-drop text file upload
   - Voice input for text (Speech-to-Text)
   - Preview mode before generation
   - Batch text processing interface

3. **Accessibility Features**
   - Keyboard navigation support
   - Screen reader compatibility
   - High contrast mode
   - Subtitle/caption support

4. **User Preferences**
   - Save favorite languages
   - Custom avatar selection
   - Voice speed preferences
   - Export format options (MP4, WebM, etc.)

#### UX Components to Design
```javascript
// Advanced Video Player Component
const AdvancedVideoPlayer = ({ videoSrc, metadata }) => {
  return (
    <div className="video-player-container">
      <video 
        controls 
        className="main-video"
        onTimeUpdate={handleTimeUpdate}
        onLoadedMetadata={handleMetadataLoad}
      >
        <source src={videoSrc} type="video/mp4" />
      </video>
      
      <div className="player-controls">
        <PlaybackSpeedControl />
        <VolumeControl />
        <ProgressBar />
        <FullscreenButton />
      </div>
      
      <div className="video-metadata">
        <span>Language: {metadata.language_name}</span>
        <span>Duration: {metadata.duration}</span>
        <span>Avatar: {metadata.avatar}</span>
      </div>
    </div>
  );
};
```

#### Design Considerations
- Mobile-responsive design
- Loading states and skeleton screens
- Smooth animations and transitions
- Intuitive error messaging
- Progressive disclosure of advanced features

---

## Shared Resources

### 📁 File Structure
```
TTS-main/
├── avatar_engine.py          # Main API service
├── tts.py                   # Basic TTS service
├── translation_agent.py     # Translation logic
├── avatar.py               # Streamlit demo
├── avatars/                # Avatar images
├── tts/tts_outputs/        # Generated audio files
├── results/                # Generated videos
├── API_DOCUMENTATION.md    # Complete API docs
├── TTS_API_Postman_Collection.json  # API testing
└── HANDOFF_README.md       # This file
```

### 🔧 Environment Setup
```bash
# Install dependencies
pip install fastapi uvicorn pyttsx3 gtts librosa keras tensorflow

# Start services
python avatar_engine.py  # Port 8001
python tts.py            # Port 8001 (different IP)

# Test endpoints
curl -X POST "http://192.168.0.125:8001/api/generate-and-sync" \
     -H "Content-Type: application/x-www-form-urlencoded" \
     -d "text=Hello World&target_lang=en"
```

### 📊 Supported Languages
- **Indian Languages**: Hindi, Marathi, Tamil, Telugu, Kannada, Malayalam, Gujarati, Bengali, Punjabi
- **International**: English, Spanish, French, German, Chinese, Japanese, Russian, Arabic, Portuguese, Italian

### 🔗 Integration Points
- **Vedant ↔ Rishabh**: API client libraries and response handling
- **Rishabh ↔ Shashank**: Component props and state management
- **Vedant ↔ Shashank**: Metadata and configuration APIs

---

## Next Steps & Timeline

### Week 1
- **Vedant**: Service consolidation and database setup
- **Rishabh**: Basic React components and API integration
- **Shashank**: UX wireframes and component design

### Week 2
- **Vedant**: Performance optimization and caching
- **Rishabh**: State management and error handling
- **Shashank**: Advanced media controls implementation

### Week 3
- **All**: Integration testing and bug fixes
- **All**: Performance testing and optimization
- **All**: Documentation updates

### Week 4
- **All**: Final testing and deployment preparation
- **All**: Demo preparation and handoff documentation

---

## Support & Communication

- **Technical Questions**: Use project Slack channel
- **Code Reviews**: Create PRs for peer review
- **Daily Standups**: 10 AM daily sync
- **Weekly Demo**: Friday 3 PM progress showcase

## Success Metrics

- ✅ API response time < 2 seconds for audio generation
- ✅ Video generation time < 30 seconds for 30-second clips
- ✅ 99% uptime for production API
- ✅ Support for all 19 languages with >90% translation accuracy
- ✅ Mobile-responsive UI with <3 second load time

---

**Good luck with the implementation! 🚀**
