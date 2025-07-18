"""
Lesson Content Management System for TTS Integration
Handles lesson structure, asset mapping, and team handoff coordination
"""

import json
import os
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LessonManager:
    """Manages lesson content structure and asset mapping"""
    
    def __init__(self, lessons_dir: str = "lessons"):
        self.lessons_dir = lessons_dir
        self.lessons_index_file = os.path.join(lessons_dir, "lessons_index.json")
        os.makedirs(lessons_dir, exist_ok=True)
        
        # Initialize lessons index if it doesn't exist
        if not os.path.exists(self.lessons_index_file):
            self._create_lessons_index()
    
    def _create_lessons_index(self):
        """Create initial lessons index structure"""
        index_structure = {
            "version": "1.0.0",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "total_lessons": 0,
            "lessons": {},
            "categories": {
                "science": [],
                "math": [],
                "language": [],
                "history": [],
                "general": []
            },
            "metadata": {
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "total_assets": 0,
                "cloud_storage_enabled": True
            }
        }
        
        with open(self.lessons_index_file, 'w') as f:
            json.dump(index_structure, f, indent=2)
        
        logger.info(f"✅ Created lessons index at {self.lessons_index_file}")
    
    def create_lesson(self, 
                     title: str, 
                     content: str, 
                     category: str = "general",
                     target_age: str = "elementary",
                     difficulty: str = "beginner",
                     duration_minutes: int = 5) -> str:
        """Create a new lesson and return lesson ID"""
        
        lesson_id = f"lesson_{str(uuid.uuid4())[:8]}"
        
        lesson_data = {
            "lesson_id": lesson_id,
            "title": title,
            "content": content,
            "category": category,
            "target_age": target_age,
            "difficulty": difficulty,
            "duration_minutes": duration_minutes,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "draft",
            "assets": {
                "audio_url": None,
                "video_url": None,
                "metadata_url": None,
                "sync_map": None
            },
            "tts_config": {
                "emotion": "balanced",
                "speed": 1.0,
                "pitch": 0.0,
                "voice_style": "educational"
            },
            "team_assignments": {
                "content_review": "akash",
                "ui_integration": "rishabh", 
                "api_endpoints": "vedant",
                "visual_sync": "shashank"
            },
            "production_status": {
                "content_approved": False,
                "assets_generated": False,
                "cloud_uploaded": False,
                "ui_integrated": False,
                "qa_tested": False
            }
        }
        
        # Save lesson file
        lesson_file = os.path.join(self.lessons_dir, f"{lesson_id}.json")
        with open(lesson_file, 'w') as f:
            json.dump(lesson_data, f, indent=2)
        
        # Update lessons index
        self._update_lessons_index(lesson_id, lesson_data)
        
        logger.info(f"✅ Created lesson {lesson_id}: {title}")
        return lesson_id
    
    def _update_lessons_index(self, lesson_id: str, lesson_data: Dict[str, Any]):
        """Update the lessons index with new lesson"""
        with open(self.lessons_index_file, 'r') as f:
            index = json.load(f)
        
        # Add lesson to index
        index["lessons"][lesson_id] = {
            "title": lesson_data["title"],
            "category": lesson_data["category"],
            "status": lesson_data["status"],
            "created_at": lesson_data["created_at"],
            "assets_ready": bool(lesson_data["assets"]["audio_url"])
        }
        
        # Update category
        category = lesson_data["category"]
        if category in index["categories"]:
            index["categories"][category].append(lesson_id)
        else:
            index["categories"]["general"].append(lesson_id)
        
        # Update metadata
        index["total_lessons"] = len(index["lessons"])
        index["metadata"]["last_updated"] = datetime.now(timezone.utc).isoformat()
        
        with open(self.lessons_index_file, 'w') as f:
            json.dump(index, f, indent=2)
    
    def update_lesson_assets(self, lesson_id: str, assets: Dict[str, str], sync_map: Optional[Dict] = None):
        """Update lesson with generated assets and sync map"""
        lesson_file = os.path.join(self.lessons_dir, f"{lesson_id}.json")
        
        if not os.path.exists(lesson_file):
            raise ValueError(f"Lesson {lesson_id} not found")
        
        with open(lesson_file, 'r') as f:
            lesson_data = json.load(f)
        
        # Update assets
        lesson_data["assets"].update(assets)
        
        # Add sync map if provided
        if sync_map:
            lesson_data["assets"]["sync_map"] = sync_map
        
        # Update production status
        lesson_data["production_status"]["assets_generated"] = True
        lesson_data["production_status"]["cloud_uploaded"] = bool(assets.get("audio_url"))
        lesson_data["status"] = "assets_ready"
        lesson_data["last_updated"] = datetime.now(timezone.utc).isoformat()
        
        # Save updated lesson
        with open(lesson_file, 'w') as f:
            json.dump(lesson_data, f, indent=2)
        
        # Update index
        self._update_lessons_index(lesson_id, lesson_data)
        
        logger.info(f"✅ Updated assets for lesson {lesson_id}")
    
    def get_lesson(self, lesson_id: str) -> Optional[Dict[str, Any]]:
        """Get lesson data by ID"""
        lesson_file = os.path.join(self.lessons_dir, f"{lesson_id}.json")
        
        if not os.path.exists(lesson_file):
            return None
        
        with open(lesson_file, 'r') as f:
            return json.load(f)
    
    def get_lessons_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Get all lessons in a category"""
        with open(self.lessons_index_file, 'r') as f:
            index = json.load(f)
        
        lesson_ids = index["categories"].get(category, [])
        lessons = []
        
        for lesson_id in lesson_ids:
            lesson_data = self.get_lesson(lesson_id)
            if lesson_data:
                lessons.append(lesson_data)
        
        return lessons
    
    def get_lessons_index(self) -> Dict[str, Any]:
        """Get the complete lessons index"""
        with open(self.lessons_index_file, 'r') as f:
            return json.load(f)
    
    def create_sample_lessons(self) -> List[str]:
        """Create 4 sample lessons for team testing"""
        sample_lessons = [
            {
                "title": "Introduction to Solar System",
                "content": "Welcome to our exciting journey through space! Today we'll explore the amazing solar system with its eight planets, bright stars, and mysterious moons. Get ready to discover the wonders of our cosmic neighborhood!",
                "category": "science",
                "target_age": "elementary",
                "difficulty": "beginner"
            },
            {
                "title": "Basic Addition and Subtraction",
                "content": "Let's learn the magic of numbers! Addition means putting things together to make more, while subtraction means taking things away. We'll practice with fun examples using toys, fruits, and animals!",
                "category": "math", 
                "target_age": "elementary",
                "difficulty": "beginner"
            },
            {
                "title": "Story Time: The Brave Little Mouse",
                "content": "Once upon a time, there lived a brave little mouse named Pip. Despite being small, Pip had a big heart and always helped friends in need. Today we'll learn about courage, friendship, and believing in yourself!",
                "category": "language",
                "target_age": "elementary", 
                "difficulty": "beginner"
            },
            {
                "title": "Ancient Egypt and Pyramids",
                "content": "Travel back in time to ancient Egypt! Discover how the mighty pyramids were built, meet the pharaohs who ruled the land, and learn about the fascinating culture along the Nile River. History comes alive!",
                "category": "history",
                "target_age": "middle_school",
                "difficulty": "intermediate"
            }
        ]
        
        created_lesson_ids = []
        for lesson in sample_lessons:
            lesson_id = self.create_lesson(**lesson)
            created_lesson_ids.append(lesson_id)
        
        logger.info(f"✅ Created {len(created_lesson_ids)} sample lessons")
        return created_lesson_ids

# Global lesson manager instance
lesson_manager = LessonManager()
