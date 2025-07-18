# 🚀 TTS Integration Deployment & Setup Guide

## 📋 Environment Setup

### Prerequisites
```bash
# Python 3.8+ required
python --version

# Install system dependencies (Windows)
# Download and install Microsoft Visual C++ Build Tools
# https://visualstudio.microsoft.com/visual-cpp-build-tools/

# Install FFmpeg
# Download from: https://ffmpeg.org/download.html
# Add to system PATH
```

### Python Dependencies
```bash
# Core dependencies
pip install fastapi uvicorn pyttsx3 gtts librosa soundfile
pip install keras tensorflow numpy pandas
pip install boto3 aiofiles asyncio
pip install pydub requests

# LoRA TTS dependencies (optional)
pip install -r requirements_lora_tts.txt

# Multimodal sentiment analysis
pip install torch transformers scikit-learn
```

---

## 🔧 Configuration

### Environment Variables
Create a `.env` file in the project root:

```bash
# AWS S3 Configuration
AWS_ACCESS_KEY_ID=your_access_key_here
AWS_SECRET_ACCESS_KEY=your_secret_key_here
AWS_S3_BUCKET=tts-assets-bucket
AWS_REGION=us-east-1
CDN_BASE_URL=https://your-cdn-domain.com

# API Configuration
API_HOST=0.0.0.0
API_PORT=8002
DEBUG_MODE=false

# Audio Configuration
AUDIO_COMPRESSION_QUALITY=128
ENABLE_TRANSITION_TONES=true
TARGET_SAMPLE_RATE=22050

# Python Configuration
PYTHONIOENCODING=utf-8
PYTHONUTF8=1
```

### AWS S3 Bucket Setup
```bash
# Create S3 bucket
aws s3 mb s3://tts-assets-bucket --region us-east-1

# Set bucket policy for public read access
aws s3api put-bucket-policy --bucket tts-assets-bucket --policy '{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "PublicReadGetObject",
      "Effect": "Allow",
      "Principal": "*",
      "Action": "s3:GetObject",
      "Resource": "arn:aws:s3:::tts-assets-bucket/*"
    }
  ]
}'

# Enable CORS for web access
aws s3api put-bucket-cors --bucket tts-assets-bucket --cors-configuration '{
  "CORSRules": [
    {
      "AllowedHeaders": ["*"],
      "AllowedMethods": ["GET", "HEAD"],
      "AllowedOrigins": ["*"],
      "MaxAgeSeconds": 3000
    }
  ]
}'
```

---

## 🏃‍♂️ Running the System

### Development Mode
```bash
# Start the TTS API server
python avatar_engine.py

# Server will start on: http://localhost:8002
# Health check: http://localhost:8002/
```

### Production Mode
```bash
# Using Gunicorn (recommended for production)
pip install gunicorn

# Start with multiple workers
gunicorn -w 4 -k uvicorn.workers.UvicornWorker avatar_engine:app --bind 0.0.0.0:8002

# Or using Docker
docker build -t tts-integration .
docker run -p 8002:8002 --env-file .env tts-integration
```

### Docker Setup
Create `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p tts/tts_outputs results sync_maps lessons

# Expose port
EXPOSE 8002

# Start the application
CMD ["python", "avatar_engine.py"]
```

Create `docker-compose.yml`:
```yaml
version: '3.8'

services:
  tts-api:
    build: .
    ports:
      - "8002:8002"
    environment:
      - AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
      - AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
      - AWS_S3_BUCKET=${AWS_S3_BUCKET}
      - AWS_REGION=${AWS_REGION}
    volumes:
      - ./tts:/app/tts
      - ./results:/app/results
      - ./sync_maps:/app/sync_maps
      - ./lessons:/app/lessons
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - tts-api
    restart: unless-stopped
```

---

## 🌐 Production Deployment

### Nginx Configuration
Create `nginx.conf`:
```nginx
events {
    worker_connections 1024;
}

http {
    upstream tts_backend {
        server tts-api:8002;
    }

    server {
        listen 80;
        server_name your-domain.com;

        # Redirect HTTP to HTTPS
        return 301 https://$server_name$request_uri;
    }

    server {
        listen 443 ssl http2;
        server_name your-domain.com;

        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;

        # API endpoints
        location /api/ {
            proxy_pass http://tts_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # Increase timeouts for video generation
            proxy_connect_timeout 300s;
            proxy_send_timeout 300s;
            proxy_read_timeout 300s;
        }

        # Static file serving
        location /static/ {
            alias /app/static/;
            expires 1y;
            add_header Cache-Control "public, immutable";
        }

        # Health check
        location /health {
            proxy_pass http://tts_backend/;
        }
    }
}
```

### SSL Certificate Setup
```bash
# Using Let's Encrypt (free SSL)
sudo apt install certbot python3-certbot-nginx

# Generate certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

---

## 📊 Monitoring & Logging

### Health Checks
```bash
# API health check
curl http://localhost:8002/

# Sentiment analysis health
curl http://localhost:8002/api/sentiment-health

# TTS configuration check
curl http://localhost:8002/api/tts-config
```

### Log Configuration
Add to `avatar_engine.py`:
```python
import logging
from logging.handlers import RotatingFileHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler('tts_api.log', maxBytes=10485760, backupCount=5),
        logging.StreamHandler()
    ]
)
```

### Monitoring Script
Create `monitor.py`:
```python
import requests
import time
import logging
from datetime import datetime

def health_check():
    try:
        response = requests.get('http://localhost:8002/', timeout=10)
        if response.status_code == 200:
            print(f"✅ {datetime.now()}: API is healthy")
            return True
        else:
            print(f"❌ {datetime.now()}: API returned {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ {datetime.now()}: API health check failed: {e}")
        return False

if __name__ == "__main__":
    while True:
        health_check()
        time.sleep(60)  # Check every minute
```

---

## 🔒 Security Configuration

### API Rate Limiting
```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add to endpoints
@app.post("/api/generate-and-sync")
@limiter.limit("10/minute")
async def generate_and_sync(request: Request, ...):
    # endpoint code
```

### Input Validation
```python
from pydantic import BaseModel, validator

class LessonRequest(BaseModel):
    title: str
    content: str
    category: str = "general"
    
    @validator('title')
    def title_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError('Title cannot be empty')
        return v
    
    @validator('content')
    def content_length_check(cls, v):
        if len(v) > 5000:
            raise ValueError('Content too long (max 5000 characters)')
        return v
```

---

## 🚀 Performance Optimization

### Caching Strategy
```python
import redis
from functools import wraps

# Redis cache setup
redis_client = redis.Redis(host='localhost', port=6379, db=0)

def cache_result(expiration=3600):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # Try to get from cache
            cached = redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
            
            # Generate and cache result
            result = await func(*args, **kwargs)
            redis_client.setex(cache_key, expiration, json.dumps(result))
            return result
        return wrapper
    return decorator
```

### Database Optimization
```python
# Use SQLite for metadata storage
import sqlite3
import json

class MetadataDB:
    def __init__(self, db_path="tts_metadata.db"):
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()
    
    def save_metadata(self, session_id, metadata):
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT OR REPLACE INTO sessions (session_id, metadata) VALUES (?, ?)",
            (session_id, json.dumps(metadata))
        )
        conn.commit()
        conn.close()
```

---

## 📈 Scaling Considerations

### Load Balancing
```yaml
# docker-compose.yml for multiple instances
version: '3.8'

services:
  tts-api-1:
    build: .
    environment:
      - INSTANCE_ID=1
  
  tts-api-2:
    build: .
    environment:
      - INSTANCE_ID=2
  
  nginx:
    image: nginx:alpine
    volumes:
      - ./nginx-lb.conf:/etc/nginx/nginx.conf
    ports:
      - "80:80"
    depends_on:
      - tts-api-1
      - tts-api-2
```

### Queue System
```python
import celery

# Celery configuration for background tasks
app = celery.Celery('tts_tasks', broker='redis://localhost:6379')

@app.task
def generate_lesson_assets_async(lesson_id):
    # Background asset generation
    pass
```

---

## 🎉 Deployment Checklist

- [ ] Environment variables configured
- [ ] AWS S3 bucket created and configured
- [ ] SSL certificates installed
- [ ] Nginx reverse proxy configured
- [ ] Health monitoring setup
- [ ] Log rotation configured
- [ ] Backup strategy implemented
- [ ] Rate limiting enabled
- [ ] Security headers configured
- [ ] Performance monitoring active

**🚀 Your TTS integration is ready for production deployment!**
