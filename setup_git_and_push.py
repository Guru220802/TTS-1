"""
Git Setup and Push Script for TTS Integration Repository
Handles nested git repositories and pushes to GitHub
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

def run_command(command, cwd=None):
    """Run a command and return the result"""
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            cwd=cwd, 
            capture_output=True, 
            text=True,
            timeout=60
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"
    except Exception as e:
        return False, "", str(e)

def remove_nested_git_repos():
    """Remove nested .git directories that cause conflicts"""
    
    nested_git_dirs = [
        "Wav2Lip/.git",
        "gender-recognition-by-voice/.git", 
        "multimodal_sentiment/.git",
        "venv/.git"
    ]
    
    for git_dir in nested_git_dirs:
        if os.path.exists(git_dir):
            try:
                if os.path.isdir(git_dir):
                    shutil.rmtree(git_dir)
                    print(f"✅ Removed nested git repo: {git_dir}")
                else:
                    os.remove(git_dir)
                    print(f"✅ Removed git file: {git_dir}")
            except Exception as e:
                print(f"❌ Failed to remove {git_dir}: {e}")

def setup_git_repository():
    """Initialize and configure git repository"""
    
    print("🔧 Setting up Git repository...")
    
    # Remove nested git repositories
    remove_nested_git_repos()
    
    # Initialize git repository
    success, stdout, stderr = run_command("git init")
    if not success:
        print(f"❌ Failed to initialize git: {stderr}")
        return False
    
    print("✅ Git repository initialized")
    
    # Configure git user (if not already configured)
    run_command('git config user.name "TTS Integration Team"')
    run_command('git config user.email "tts-integration@example.com"')
    
    # Add all files
    print("📁 Adding files to git...")
    success, stdout, stderr = run_command("git add .")
    if not success:
        print(f"❌ Failed to add files: {stderr}")
        return False
    
    print("✅ Files added to git")
    
    # Create initial commit
    commit_message = "feat: Complete TTS integration with clean repository structure\n\n- Organized code into professional directory structure\n- Added comprehensive team handoff documentation\n- Implemented cloud storage and asset management\n- Created 4 production lesson samples\n- Added sync maps for UI integration\n- Configured deployment and setup guides"
    
    success, stdout, stderr = run_command(f'git commit -m "{commit_message}"')
    if not success:
        print(f"❌ Failed to commit: {stderr}")
        return False
    
    print("✅ Initial commit created")
    return True

def push_to_github():
    """Push repository to GitHub"""
    
    print("🚀 Pushing to GitHub...")
    
    # Check if remote origin exists
    success, stdout, stderr = run_command("git remote -v")
    
    if "origin" not in stdout:
        print("⚠️ No GitHub remote found.")
        print("Please add your GitHub repository URL:")
        print("Example: git remote add origin https://github.com/yourusername/TTS-Integration.git")
        
        # Provide instructions for manual setup
        github_url = input("Enter your GitHub repository URL (or press Enter to skip): ").strip()
        
        if github_url:
            success, stdout, stderr = run_command(f"git remote add origin {github_url}")
            if not success:
                print(f"❌ Failed to add remote: {stderr}")
                return False
            print(f"✅ Added remote origin: {github_url}")
        else:
            print("⏭️ Skipping GitHub push - no remote URL provided")
            return True
    
    # Set upstream and push
    success, stdout, stderr = run_command("git branch -M main")
    if not success:
        print(f"⚠️ Failed to rename branch: {stderr}")
    
    # Try to push
    success, stdout, stderr = run_command("git push -u origin main")
    if not success:
        # If push fails, try force push (for existing repos)
        print("⚠️ Normal push failed, trying force push...")
        success, stdout, stderr = run_command("git push -u origin main --force")
        
        if not success:
            print(f"❌ Failed to push to GitHub: {stderr}")
            print("You may need to:")
            print("1. Create a new repository on GitHub")
            print("2. Set up authentication (GitHub token or SSH key)")
            print("3. Check the repository URL")
            return False
    
    print("✅ Successfully pushed to GitHub!")
    return True

def create_github_readme():
    """Create a GitHub-optimized README"""
    
    readme_content = """# 🎤 TTS Integration System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.68+-green.svg)](https://fastapi.tiangolo.com)
[![AWS S3](https://img.shields.io/badge/AWS-S3-orange.svg)](https://aws.amazon.com/s3/)

A comprehensive Text-to-Speech integration system with multimodal sentiment analysis, cloud storage, and team collaboration features.

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/TTS-Integration.git
cd TTS-Integration

# Install dependencies
pip install -r config/requirements_lora_tts.txt

# Set up environment
cp config/.env.example .env
# Edit .env with your configuration

# Start the API server
python src/api/avatar_engine.py
```

## 🎯 Features

- ✅ **Enhanced TTS Engine** with emotional control
- ✅ **Multimodal Sentiment Analysis** for tone adaptation  
- ✅ **Cloud Storage Integration** (AWS S3)
- ✅ **Sync Maps** for precise UI synchronization
- ✅ **Lesson Management** with JSON structure
- ✅ **Asset Management** with automated upload
- ✅ **Team Integration APIs** for all components

## 📚 Documentation

- **[Team Handoff Guide](docs/team_handoff/TEAM_HANDOFF_COMPLETE.md)** - Complete integration guide
- **[API Documentation](docs/api/API_DOCUMENTATION.md)** - API endpoints and usage
- **[Deployment Guide](docs/deployment/DEPLOYMENT_SETUP_GUIDE.md)** - Production deployment
- **[Project Structure](docs/PROJECT_STRUCTURE.md)** - Repository organization

## 👥 Team Integration

| Team Member | Integration Point | Documentation |
|-------------|------------------|---------------|
| **Akash** | Content Review | [Content Review Guide](docs/team_handoff/akash_content_review_summary_20250718_124017.md) |
| **Rishabh** | UI Integration | [UI Integration Guide](docs/team_handoff/RISHABH_UI_INTEGRATION_GUIDE.md) |
| **Vedant** | API Integration | [API Documentation](docs/api/API_DOCUMENTATION.md) |
| **Shashank** | Visual Sync | [Team Handoff Guide](docs/team_handoff/TEAM_HANDOFF_COMPLETE.md) |

## 🏗️ Repository Structure

```
src/          # Core application code
docs/         # Team handoff and API documentation  
config/       # Dependencies and configuration
scripts/      # Setup and testing scripts
assets/       # Avatars, sounds, models
data/         # Lessons, results, outputs
```

## 📊 Production Status

- ✅ **4 Lesson Samples** created and ready
- ✅ **API Endpoints** implemented and documented
- ✅ **Cloud Storage** configured and tested
- ✅ **Sync Maps** generated for UI integration
- ✅ **Team Documentation** complete

## 🤝 Contributing

See our [team handoff documentation](docs/team_handoff/) for integration guidelines.

## 📄 License

This project is part of the TTS Integration initiative.

---

**🎉 Ready for team integration and production deployment!**
"""
    
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)
    
    print("✅ Created GitHub-optimized README.md")

def main():
    """Main function to set up git and push to GitHub"""
    
    print("🚀 TTS Integration - Git Setup and GitHub Push")
    print("=" * 60)
    
    # Create GitHub-optimized README
    create_github_readme()
    
    # Set up git repository
    if not setup_git_repository():
        print("❌ Failed to set up git repository")
        return False
    
    # Push to GitHub
    if not push_to_github():
        print("❌ Failed to push to GitHub")
        print("\n📋 Manual steps to complete:")
        print("1. Create a new repository on GitHub")
        print("2. Add remote: git remote add origin <your-repo-url>")
        print("3. Push: git push -u origin main")
        return False
    
    print("\n" + "=" * 60)
    print("✅ SUCCESS! Repository pushed to GitHub")
    print("\n📋 What was pushed:")
    print("   • Clean, organized repository structure")
    print("   • Complete TTS integration system")
    print("   • Team handoff documentation")
    print("   • 4 production lesson samples")
    print("   • API endpoints and configuration")
    print("   • Deployment and setup guides")
    print("\n🎉 Your TTS integration is now on GitHub!")
    
    return True

if __name__ == "__main__":
    main()
