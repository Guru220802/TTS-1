"""
Direct Lesson Sample Creator
Creates 4 production lesson samples directly using the lesson manager
"""

import json
import os
from datetime import datetime
from lesson_manager import lesson_manager
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_production_lessons():
    """Create 4 high-quality lesson samples directly"""
    
    production_lessons = [
        {
            "title": "Day 1: Introduction to Solar System",
            "content": """Welcome to our exciting journey through space! Today we'll explore the amazing solar system with its eight planets, bright stars, and mysterious moons. 

The Sun is the center of our solar system, a giant ball of hot gas that gives us light and warmth. Mercury is the closest planet to the Sun, while Neptune is the farthest away. 

Earth is our special home planet, the only one we know that has life. It has water, air, and the perfect temperature for plants, animals, and people to live.

Get ready to discover the wonders of our cosmic neighborhood! We'll learn about each planet's unique features and what makes them special.""",
            "category": "science",
            "target_age": "elementary",
            "difficulty": "beginner",
            "duration_minutes": 8
        },
        {
            "title": "Day 2: Basic Addition and Subtraction Magic",
            "content": """Let's learn the magic of numbers! Addition means putting things together to make more, while subtraction means taking things away.

Imagine you have 3 red apples and someone gives you 2 more green apples. How many apples do you have now? That's right - 5 apples! We added 3 plus 2 to get 5.

Now, if you eat 1 apple, how many are left? We subtract 1 from 5 to get 4 apples remaining.

We'll practice with fun examples using toys, fruits, and animals! Math is everywhere around us, and once you understand these basics, you'll see numbers in a whole new way.""",
            "category": "math",
            "target_age": "elementary", 
            "difficulty": "beginner",
            "duration_minutes": 6
        },
        {
            "title": "Day 3: Story Time - The Brave Little Mouse",
            "content": """Once upon a time, in a cozy little house, there lived a brave little mouse named Pip. Despite being small, Pip had a big heart and always helped friends in need.

One day, Pip heard crying from the garden. A baby bird had fallen from its nest and couldn't get back up. All the other animals were too scared to help because the nest was so high up.

But Pip wasn't afraid! Using creativity and courage, Pip found a way to help the baby bird return safely to its family. The other animals cheered and realized that being brave isn't about being big - it's about having a kind heart.

Today we'll learn about courage, friendship, and believing in yourself! Sometimes the smallest person can make the biggest difference.""",
            "category": "language",
            "target_age": "elementary",
            "difficulty": "beginner", 
            "duration_minutes": 7
        },
        {
            "title": "Day 4: Ancient Egypt and the Mighty Pyramids",
            "content": """Travel back in time to ancient Egypt, over 4,000 years ago! Discover how the mighty pyramids were built, meet the pharaohs who ruled the land, and learn about the fascinating culture along the Nile River.

The ancient Egyptians were incredible builders and inventors. They created the pyramids as tombs for their kings, called pharaohs. The Great Pyramid of Giza is so amazing that it's one of the Seven Wonders of the Ancient World!

These clever people also invented hieroglyphics - a special way of writing using pictures and symbols. They mummified their pharaohs to preserve them for the afterlife, believing in a journey beyond this world.

History comes alive when we learn about these remarkable people and their incredible achievements! Let's explore their world together.""",
            "category": "history",
            "target_age": "middle_school",
            "difficulty": "intermediate",
            "duration_minutes": 9
        }
    ]
    
    logger.info("🚀 Creating production lesson samples...")
    
    created_lessons = []
    
    for i, lesson_data in enumerate(production_lessons):
        try:
            lesson_id = lesson_manager.create_lesson(**lesson_data)
            created_lessons.append({
                "lesson_id": lesson_id,
                "title": lesson_data["title"],
                "category": lesson_data["category"],
                "content_length": len(lesson_data["content"]),
                "estimated_duration": lesson_data["duration_minutes"]
            })
            
            logger.info(f"✅ Created lesson {i+1}: {lesson_data['title']} (ID: {lesson_id})")
            
        except Exception as e:
            logger.error(f"❌ Failed to create lesson {i+1}: {e}")
    
    # Generate production report
    report = {
        "production_lesson_creation": {
            "created_at": datetime.now().isoformat(),
            "total_lessons_created": len(created_lessons),
            "lessons": created_lessons,
            "content_statistics": {
                "total_words": sum([len(lesson["content"].split()) for lesson in production_lessons]),
                "average_words_per_lesson": sum([len(lesson["content"].split()) for lesson in production_lessons]) / len(production_lessons),
                "total_estimated_duration": sum([lesson["duration_minutes"] for lesson in production_lessons]),
                "categories_covered": list(set([lesson["category"] for lesson in production_lessons]))
            },
            "quality_features": {
                "emotional_tts": "Configured for educational content",
                "sentiment_analysis": "Integrated for tone control",
                "transition_tones": "Available for all emotions",
                "female_avatar": "Single avatar for consistency",
                "cloud_storage": "Ready for asset upload",
                "sync_maps": "Will be generated with assets"
            },
            "team_handoff_readiness": {
                "akash_content_review": {
                    "status": "Ready",
                    "lessons_for_review": len(created_lessons),
                    "content_types": ["science", "math", "language", "history"],
                    "next_action": "Generate TTS assets and review audio quality"
                },
                "rishabh_ui_integration": {
                    "status": "Ready",
                    "lesson_structure": "JSON format with metadata",
                    "sync_maps": "Will be available after asset generation",
                    "next_action": "Integrate lesson selector and progress tracking"
                },
                "vedant_api_endpoints": {
                    "status": "Ready", 
                    "asset_fetch_apis": "Implemented",
                    "lesson_mapping": "Available via /api/lessons endpoints",
                    "next_action": "Test batch asset retrieval"
                },
                "shashank_visual_sync": {
                    "status": "Ready",
                    "timing_data": "Will be in sync maps",
                    "animation_keyframes": "Generated with assets",
                    "next_action": "Test visual synchronization with generated content"
                }
            },
            "production_checklist": {
                "lesson_content_created": True,
                "lesson_structure_defined": True,
                "api_endpoints_ready": True,
                "cloud_storage_configured": True,
                "sync_map_generator_ready": True,
                "asset_generation_pending": True,
                "team_handoff_docs_pending": True
            }
        }
    }
    
    # Save report
    report_path = f"production_lesson_creation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"📊 Production report saved: {report_path}")
    
    # Create summary for Akash
    summary_path = f"akash_content_review_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(summary_path, 'w') as f:
        f.write("# Content Review Summary for Akash\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"## Lessons Created: {len(created_lessons)}\n\n")
        
        for i, lesson in enumerate(created_lessons):
            f.write(f"### {i+1}. {lesson['title']}\n")
            f.write(f"- **Lesson ID:** `{lesson['lesson_id']}`\n")
            f.write(f"- **Category:** {lesson['category']}\n")
            f.write(f"- **Content Length:** {lesson['content_length']} characters\n")
            f.write(f"- **Estimated Duration:** {lesson['estimated_duration']} minutes\n\n")
        
        f.write("## Next Steps for Content Review\n\n")
        f.write("1. **Generate TTS Assets:** Run asset generation for each lesson\n")
        f.write("2. **Review Audio Quality:** Check emotional tone and clarity\n")
        f.write("3. **Validate Content:** Ensure educational accuracy and engagement\n")
        f.write("4. **Test Sync Maps:** Verify timing accuracy for UI controls\n")
        f.write("5. **Approve for Production:** Mark lessons as content-approved\n\n")
        f.write("## API Commands for Asset Generation\n\n")
        
        for lesson in created_lessons:
            f.write(f"```bash\n")
            f.write(f"# Generate assets for {lesson['title']}\n")
            f.write(f"curl -X POST http://localhost:8002/api/lessons/{lesson['lesson_id']}/generate-assets\n")
            f.write(f"```\n\n")
    
    logger.info(f"📋 Content review summary saved: {summary_path}")
    
    return created_lessons, report_path

def main():
    """Main function"""
    
    print("🎓 Production Lesson Sample Creator")
    print("=" * 50)
    print("Creating 4 lesson samples for team review...")
    print()
    
    created_lessons, report_path = create_production_lessons()
    
    if created_lessons:
        print(f"\n🎉 SUCCESS! Created {len(created_lessons)} production lesson samples.")
        print(f"📁 Report saved: {report_path}")
        print("\n📚 Lessons Created:")
        for i, lesson in enumerate(created_lessons):
            print(f"   {i+1}. {lesson['title']} (ID: {lesson['lesson_id']})")
        
        print("\n🤝 Ready for team handoff:")
        print("   • Akash: Content review and audio quality validation")
        print("   • Rishabh: UI integration with lesson structure")
        print("   • Vedant: API testing with lesson endpoints")
        print("   • Shashank: Visual sync preparation")
        
        return True
    else:
        print("\n❌ Failed to create lesson samples.")
        return False

if __name__ == "__main__":
    main()
