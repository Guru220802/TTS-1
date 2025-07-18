# 🎤 TTS Integration System

## 📋 Project Overview

A comprehensive Text-to-Speech integration system with multimodal sentiment analysis, cloud storage, and team collaboration features.

## 🏗️ Repository Structure

```
TTS-main/
├── src/                          # Core application code
│   ├── api/                      # API endpoints and services
│   │   ├── avatar_engine.py      # Main TTS API service
│   │   ├── lesson_manager.py     # Lesson management system
│   │   ├── sync_map_generator.py # Sync map generation
│   │   └── tts.py               # Basic TTS service
│   ├── tts/                     # TTS engine components
│   │   ├── lora_tts_engine.py   # LoRA TTS implementation
│   │   ├── translation_agent.py  # Multi-language support
│   │   └── emotional_fallback_tts.py # Fallback TTS
│   └── utils/                   # Utility functions
│
├── docs/                        # Documentation
│   ├── team_handoff/           # Team integration guides
│   ├── api/                    # API documentation
│   └── deployment/             # Deployment guides
│
├── config/                      # Configuration files
├── scripts/                     # Utility scripts
├── assets/                      # Static assets (avatars, sounds, models)
├── data/                        # Generated data and outputs
├── logs/                        # Application logs
│
├── Wav2Lip/                     # Lip-sync generation (external)
├── gender-recognition-by-voice/ # Voice gender detection (external)
└── multimodal_sentiment/        # Sentiment analysis (external)
```

## 🚀 Quick Start

### Prerequisites
```bash
# Install Python dependencies
pip install -r config/requirements_lora_tts.txt

# Set up environment variables
cp config/.env.example .env
# Edit .env with your configuration
```

### Running the System
```bash
# Start the main API server
python src/api/avatar_engine.py

# Server will be available at: http://localhost:8002
```

## 📚 Documentation

- **[Team Handoff Guide](docs/team_handoff/TEAM_HANDOFF_COMPLETE.md)** - Complete integration guide
- **[API Documentation](docs/api/API_DOCUMENTATION.md)** - API endpoints and usage
- **[Deployment Guide](docs/deployment/DEPLOYMENT_SETUP_GUIDE.md)** - Production deployment
- **[UI Integration](docs/team_handoff/RISHABH_UI_INTEGRATION_GUIDE.md)** - Frontend integration guide

## 🎯 Features

- ✅ **Enhanced TTS Engine** with emotional control
- ✅ **Multimodal Sentiment Analysis** for tone adaptation
- ✅ **Cloud Storage Integration** (AWS S3)
- ✅ **Sync Maps** for precise UI synchronization
- ✅ **Lesson Management** with JSON structure
- ✅ **Asset Management** with automated upload
- ✅ **Team Integration APIs** for all components

## 👥 Team Integration

| Team Member | Integration Point | Documentation |
|-------------|------------------|---------------|
| **Akash** | Content Review | [Content Review Guide](docs/team_handoff/akash_content_review_summary_20250718_124017.md) |
| **Rishabh** | UI Integration | [UI Integration Guide](docs/team_handoff/RISHABH_UI_INTEGRATION_GUIDE.md) |
| **Vedant** | API Integration | [API Documentation](docs/api/API_DOCUMENTATION.md) |
| **Shashank** | Visual Sync | [Team Handoff Guide](docs/team_handoff/TEAM_HANDOFF_COMPLETE.md) |

## 🧪 Testing

```bash
# Run lesson creation test
python scripts/create_sample_lessons.py

# Test TTS generation
python scripts/testing/test_emotional_tts.py

# API health check
curl http://localhost:8002/
```

## 📊 Production Status

- ✅ **4 Lesson Samples** created and ready
- ✅ **API Endpoints** implemented and documented
- ✅ **Cloud Storage** configured and tested
- ✅ **Sync Maps** generated for UI integration
- ✅ **Team Documentation** complete

## 🔧 Configuration

Key configuration files:
- `config/requirements_lora_tts.txt` - Python dependencies
- `config/TTS_API_Postman_Collection.json` - API testing collection
- `.env` - Environment variables (create from template)

## 📞 Support

- **Technical Issues**: Check logs in `logs/` directory
- **API Problems**: Review `docs/api/API_DOCUMENTATION.md`
- **Integration Help**: See team-specific guides in `docs/team_handoff/`

---

**🎉 Ready for team integration and production deployment!**
