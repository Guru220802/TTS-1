# 📁 Project Structure

## 🏗️ Organized Repository Layout

```
TTS-main/
├── 📁 src/                          # Core application code
│   ├── 📁 api/                      # API endpoints and services
│   │   ├── avatar_engine.py         # Main TTS API service
│   │   ├── lesson_manager.py        # Lesson management system
│   │   ├── sync_map_generator.py    # Sync map generation
│   │   └── tts.py                  # Basic TTS service
│   ├── 📁 tts/                     # TTS engine components
│   │   ├── lora_tts_engine.py      # LoRA TTS implementation
│   │   ├── translation_agent.py    # Multi-language support
│   │   ├── emotional_fallback_tts.py # Fallback TTS
│   │   └── lora_emotional_trainer.py # LoRA training
│   ├── 📁 sentiment/               # Multimodal sentiment analysis
│   │   └── (multimodal_sentiment moved here)
│   └── 📁 utils/                   # Utility functions
│       ├── setup_ffmpeg_env.py     # FFmpeg setup
│       └── avatar.py               # Streamlit demo
│
├── 📁 docs/                        # Documentation
│   ├── 📁 team_handoff/           # Team integration guides
│   │   ├── TEAM_HANDOFF_COMPLETE.md
│   │   ├── RISHABH_UI_INTEGRATION_GUIDE.md
│   │   ├── HANDOFF_README.md
│   │   └── akash_content_review_summary_*.md
│   ├── 📁 api/                    # API documentation
│   │   └── API_DOCUMENTATION.md
│   ├── 📁 deployment/             # Deployment guides
│   │   └── DEPLOYMENT_SETUP_GUIDE.md
│   ├── COMPLETION_SUMMARY.md       # Project completion
│   ├── FINAL_README.md            # Final documentation
│   └── TEST_CASES_MULTILINGUAL.md # Test cases
│
├── 📁 config/                      # Configuration files
│   ├── requirements_lora_tts.txt   # Python dependencies
│   ├── TTS_API_Postman_Collection.json # API testing
│   └── .env.example               # Environment template
│
├── 📁 scripts/                     # Utility scripts
│   ├── 📁 setup/                  # Setup scripts
│   │   ├── setup_emotional_tts.py
│   │   └── train_emotional_lora.py
│   ├── 📁 testing/                # Test scripts
│   │   └── test_emotional_tts.py
│   ├── create_sample_lessons.py   # Lesson creation
│   └── generate_production_lessons.py # Production generation
│
├── 📁 assets/                      # Static assets
│   ├── 📁 avatars/                # Avatar images
│   │   ├── pht2.jpg               # Primary female avatar
│   │   └── download.jpg           # Additional avatar
│   ├── 📁 transition_sounds/      # Emotion-based tones
│   │   ├── balanced_tone.wav
│   │   ├── joyful_tone.wav
│   │   └── ... (8 more emotions)
│   ├── gender_model.pkl           # Gender detection model
│   └── gender_scaler.pkl          # Model scaler
│
├── 📁 data/                        # Generated data and outputs
│   ├── 📁 lessons/                # Lesson JSON files
│   │   ├── lesson_*.json          # Individual lessons
│   │   └── lessons_index.json     # Lesson index
│   ├── 📁 results/                # Generated videos
│   │   ├── *.mp4                  # Video outputs
│   │   └── metadata_*.json        # Video metadata
│   ├── 📁 sync_maps/              # Synchronization data
│   │   └── sync_map_*.json        # Timing data for UI
│   └── 📁 tts_outputs/            # Audio processing
│       ├── *_base.mp3             # Original audio
│       ├── *_optimized.mp3        # Enhanced audio
│       └── *.wav                  # WAV for lip-sync
│
├── 📁 logs/                        # Application logs
│   └── production_lesson_creation_report_*.json
│
├── 📁 core/                        # External core systems
│   ├── 📁 wav2lip/                # Lip-sync generation
│   └── 📁 gender_recognition/     # Voice gender detection
│
├── 📁 Wav2Lip/                     # Lip-sync generation (external)
├── 📁 gender-recognition-by-voice/ # Gender detection (external)
├── 📁 multimodal_sentiment/        # Sentiment analysis (external)
│
├── README.md                       # Main project documentation
├── .gitignore                     # Git ignore rules
└── .env                           # Environment variables (create from template)
```

## 🎯 Directory Purposes

### `src/` - Core Application Code
- **`api/`** - FastAPI endpoints and main services
- **`tts/`** - Text-to-speech engine components
- **`sentiment/`** - Sentiment analysis integration
- **`utils/`** - Utility functions and helpers

### `docs/` - Documentation
- **`team_handoff/`** - Integration guides for team members
- **`api/`** - API documentation and references
- **`deployment/`** - Production deployment guides

### `config/` - Configuration
- Python dependencies, API collections, environment templates

### `scripts/` - Automation
- **`setup/`** - Installation and setup scripts
- **`testing/`** - Test and validation scripts
- Lesson creation and production scripts

### `assets/` - Static Resources
- **`avatars/`** - Avatar images for video generation
- **`transition_sounds/`** - Emotion-based audio tones
- Pre-trained models and scalers

### `data/` - Generated Content
- **`lessons/`** - Lesson content and structure
- **`results/`** - Generated videos and metadata
- **`sync_maps/`** - UI synchronization data
- **`tts_outputs/`** - Audio processing pipeline

### `logs/` - Application Logs
- Generation reports and system logs

## 🚀 Getting Started

1. **Install dependencies:**
   ```bash
   pip install -r config/requirements_lora_tts.txt
   ```

2. **Set up environment:**
   ```bash
   cp config/.env.example .env
   # Edit .env with your configuration
   ```

3. **Start the API:**
   ```bash
   python src/api/avatar_engine.py
   ```

4. **Access documentation:**
   - Main guide: `docs/team_handoff/TEAM_HANDOFF_COMPLETE.md`
   - API docs: `docs/api/API_DOCUMENTATION.md`
   - Deployment: `docs/deployment/DEPLOYMENT_SETUP_GUIDE.md`

## 📊 Integration Status

- ✅ **Core TTS Pipeline** - Fully functional
- ✅ **4 Lesson Samples** - Ready for review
- ✅ **Team APIs** - Implemented and documented
- ✅ **Cloud Storage** - Configured and tested
- ✅ **Documentation** - Complete for all team members

**🎉 Ready for team integration and production deployment!**
