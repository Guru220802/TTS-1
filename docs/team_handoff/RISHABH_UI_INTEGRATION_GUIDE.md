# 🎨 Rishabh - UI Integration Guide

## 🎯 Your Integration Scope

You're responsible for creating the frontend components that interact with the TTS system, including lesson selection, playback controls, and real-time text highlighting.

---

## ✅ What's Ready for You

### 📊 Sync Maps with UI Controls
- **Word-level timestamps** for text highlighting
- **Progress markers** for scrubbing controls
- **Playback speed options** (0.5x to 2x)
- **Skip forward/backward** controls (10-second jumps)

### 📚 Lesson Structure
- **JSON format** with complete metadata
- **Category organization** (science, math, language, history)
- **Difficulty levels** and target age groups
- **Asset URLs** for audio/video streaming

### 🔗 API Endpoints Ready
All endpoints tested and documented with example responses.

---

## 🛠️ API Integration

### Get All Lessons
```javascript
// Fetch lessons index
const response = await fetch('/api/lessons');
const lessonsIndex = await response.json();

// Structure:
{
  "total_lessons": 4,
  "lessons": {
    "lesson_34af1a27": {
      "title": "Day 1: Introduction to Solar System",
      "category": "science",
      "status": "draft",
      "assets_ready": false
    }
  },
  "categories": {
    "science": ["lesson_34af1a27"],
    "math": ["lesson_20a9eaff"]
  }
}
```

### Get Specific Lesson
```javascript
// Fetch lesson details
const lessonId = "lesson_34af1a27";
const response = await fetch(`/api/lessons/${lessonId}`);
const lesson = await response.json();

// Structure:
{
  "lesson_id": "lesson_34af1a27",
  "title": "Day 1: Introduction to Solar System",
  "content": "Welcome to our exciting journey...",
  "category": "science",
  "assets": {
    "audio_url": "https://cdn.example.com/audio/abc123.mp3",
    "video_url": "https://cdn.example.com/video/abc123.mp4"
  },
  "tts_config": {
    "emotion": "enthusiastic",
    "speed": 1.0,
    "pitch": 0.0
  }
}
```

### Get Sync Map for Playback Controls
```javascript
// Fetch sync map for a session
const sessionId = "abc123";
const response = await fetch(`/api/sync-map/${sessionId}`);
const syncMap = await response.json();

// Structure:
{
  "session_id": "abc123",
  "duration_seconds": 45.2,
  "word_timestamps": [
    {
      "word": "Welcome",
      "start_time": 0.0,
      "end_time": 0.5,
      "index": 0
    },
    {
      "word": "to",
      "start_time": 0.5,
      "end_time": 0.7,
      "index": 1
    }
  ],
  "ui_controls": {
    "progress_markers": [
      {
        "time": 0.0,
        "progress_percent": 0,
        "label": "Welcome to our exciting journey..."
      }
    ],
    "word_highlights": [...],
    "playback_controls": {
      "skip_forward": 10,
      "skip_backward": 10,
      "speed_options": [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    }
  }
}
```

---

## 🎮 React Component Examples

### Lesson Selector Component
```jsx
import React, { useState, useEffect } from 'react';

const LessonSelector = ({ onLessonSelect }) => {
  const [lessons, setLessons] = useState({});
  const [selectedCategory, setSelectedCategory] = useState('all');

  useEffect(() => {
    fetch('/api/lessons')
      .then(res => res.json())
      .then(data => setLessons(data));
  }, []);

  const filteredLessons = selectedCategory === 'all' 
    ? Object.values(lessons.lessons || {})
    : lessons.categories?.[selectedCategory]?.map(id => lessons.lessons[id]) || [];

  return (
    <div className="lesson-selector">
      <div className="category-filter">
        <select 
          value={selectedCategory} 
          onChange={(e) => setSelectedCategory(e.target.value)}
        >
          <option value="all">All Categories</option>
          <option value="science">Science</option>
          <option value="math">Math</option>
          <option value="language">Language</option>
          <option value="history">History</option>
        </select>
      </div>
      
      <div className="lesson-grid">
        {filteredLessons.map(lesson => (
          <div 
            key={lesson.lesson_id}
            className="lesson-card"
            onClick={() => onLessonSelect(lesson)}
          >
            <h3>{lesson.title}</h3>
            <span className="category">{lesson.category}</span>
            <span className={`status ${lesson.status}`}>
              {lesson.assets_ready ? '✅ Ready' : '⏳ Pending'}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
};
```

### Text Highlighting Component
```jsx
import React, { useState, useEffect } from 'react';

const TextHighlighter = ({ text, syncMap, currentTime }) => {
  const [highlightedWordIndex, setHighlightedWordIndex] = useState(-1);

  useEffect(() => {
    if (!syncMap?.word_timestamps) return;

    const currentWord = syncMap.word_timestamps.find(word => 
      currentTime >= word.start_time && currentTime <= word.end_time
    );

    setHighlightedWordIndex(currentWord ? currentWord.index : -1);
  }, [currentTime, syncMap]);

  const renderHighlightedText = () => {
    if (!syncMap?.word_timestamps) {
      return <p>{text}</p>;
    }

    const words = text.split(' ');
    
    return (
      <p className="highlighted-text">
        {words.map((word, index) => (
          <span
            key={index}
            className={`word ${index === highlightedWordIndex ? 'highlighted' : ''}`}
            data-word-index={index}
          >
            {word}{' '}
          </span>
        ))}
      </p>
    );
  };

  return (
    <div className="text-highlighter">
      {renderHighlightedText()}
    </div>
  );
};
```

### Video Player with Controls
```jsx
import React, { useState, useRef, useEffect } from 'react';

const TTSVideoPlayer = ({ lesson, syncMap }) => {
  const videoRef = useRef(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(1.0);

  const handleTimeUpdate = () => {
    if (videoRef.current) {
      setCurrentTime(videoRef.current.currentTime);
    }
  };

  const handlePlayPause = () => {
    if (videoRef.current) {
      if (isPlaying) {
        videoRef.current.pause();
      } else {
        videoRef.current.play();
      }
      setIsPlaying(!isPlaying);
    }
  };

  const handleSpeedChange = (speed) => {
    if (videoRef.current) {
      videoRef.current.playbackRate = speed;
      setPlaybackSpeed(speed);
    }
  };

  const handleSkip = (seconds) => {
    if (videoRef.current) {
      videoRef.current.currentTime += seconds;
    }
  };

  return (
    <div className="tts-video-player">
      <video
        ref={videoRef}
        src={lesson.assets.video_url}
        onTimeUpdate={handleTimeUpdate}
        onPlay={() => setIsPlaying(true)}
        onPause={() => setIsPlaying(false)}
        className="main-video"
      />
      
      <div className="player-controls">
        <button onClick={() => handleSkip(-10)}>⏪ 10s</button>
        <button onClick={handlePlayPause}>
          {isPlaying ? '⏸️' : '▶️'}
        </button>
        <button onClick={() => handleSkip(10)}>⏩ 10s</button>
        
        <div className="speed-control">
          <label>Speed:</label>
          <select 
            value={playbackSpeed} 
            onChange={(e) => handleSpeedChange(parseFloat(e.target.value))}
          >
            <option value={0.5}>0.5x</option>
            <option value={0.75}>0.75x</option>
            <option value={1.0}>1.0x</option>
            <option value={1.25}>1.25x</option>
            <option value={1.5}>1.5x</option>
            <option value={2.0}>2.0x</option>
          </select>
        </div>
      </div>
      
      <TextHighlighter 
        text={lesson.content}
        syncMap={syncMap}
        currentTime={currentTime}
      />
    </div>
  );
};
```

---

## 🎨 CSS Styling Examples

```css
/* Lesson Selector Styles */
.lesson-selector {
  padding: 20px;
}

.lesson-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 20px;
  margin-top: 20px;
}

.lesson-card {
  border: 1px solid #ddd;
  border-radius: 8px;
  padding: 20px;
  cursor: pointer;
  transition: all 0.3s ease;
}

.lesson-card:hover {
  box-shadow: 0 4px 12px rgba(0,0,0,0.1);
  transform: translateY(-2px);
}

/* Text Highlighting Styles */
.highlighted-text {
  font-size: 18px;
  line-height: 1.6;
  padding: 20px;
  background: #f9f9f9;
  border-radius: 8px;
}

.word {
  transition: all 0.3s ease;
  padding: 2px 4px;
  border-radius: 4px;
}

.word.highlighted {
  background-color: #ffeb3b;
  font-weight: bold;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

/* Video Player Styles */
.tts-video-player {
  max-width: 800px;
  margin: 0 auto;
}

.main-video {
  width: 100%;
  border-radius: 8px;
}

.player-controls {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 15px;
  background: #f5f5f5;
  border-radius: 8px;
  margin-top: 10px;
}

.player-controls button {
  padding: 8px 16px;
  border: none;
  border-radius: 4px;
  background: #007bff;
  color: white;
  cursor: pointer;
}

.player-controls button:hover {
  background: #0056b3;
}
```

---

## 🧪 Testing Your Integration

### 1. Test Lesson Loading
```javascript
// Test lesson fetching
async function testLessonLoading() {
  try {
    const response = await fetch('/api/lessons');
    const data = await response.json();
    console.log('Lessons loaded:', data.total_lessons);
    return data;
  } catch (error) {
    console.error('Failed to load lessons:', error);
  }
}
```

### 2. Test Sync Map Integration
```javascript
// Test sync map functionality
async function testSyncMap(sessionId) {
  try {
    const response = await fetch(`/api/sync-map/${sessionId}`);
    const syncMap = await response.json();
    console.log('Sync map loaded:', syncMap.word_timestamps.length, 'words');
    return syncMap;
  } catch (error) {
    console.error('Failed to load sync map:', error);
  }
}
```

---

## 🚀 Next Steps

1. **Set up development environment** with the API server
2. **Implement lesson selector** component
3. **Add text highlighting** with sync map integration
4. **Create video player** with custom controls
5. **Test with generated lesson samples**
6. **Optimize for mobile responsiveness**

---

## 📞 Support

- **API Issues**: Check server logs in `avatar_engine.py`
- **Sync Map Problems**: Verify session IDs match generated content
- **Asset Loading**: Ensure cloud URLs are accessible
- **Performance**: Implement caching for frequently accessed lessons

**🎉 Your UI integration points are ready! Start building amazing user experiences!**
