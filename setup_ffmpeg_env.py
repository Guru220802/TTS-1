import os
import sys
import subprocess
from pathlib import Path


def setup_ffmpeg_path():
    """
    Setup FFmpeg environment path for the application.
    This function attempts to locate FFmpeg and add it to the system PATH.
    """
    # Common FFmpeg installation paths on Windows
    common_paths = [
        # Chocolatey installation paths
        r"C:\ProgramData\chocolatey\lib\ffmpeg\tools\ffmpeg\bin",
        r"C:\ProgramData\chocolatey\bin",
        r"C:\tools\ffmpeg\bin",
        # Standard installation paths
        r"C:\ffmpeg\bin",
        r"C:\Program Files\ffmpeg\bin",
        r"C:\Program Files (x86)\ffmpeg\bin",
        os.path.join(os.path.expanduser("~"), "ffmpeg", "bin"),
        # Add current directory and common relative paths
        os.path.join(os.getcwd(), "ffmpeg", "bin"),
        os.path.join(os.getcwd(), "bin"),
    ]
    
    # Check if ffmpeg is already available in PATH
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, 
                              text=True, 
                              timeout=5)
        if result.returncode == 0:
            print("✅ FFmpeg is already available in PATH")
            return True
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    # Try to find FFmpeg in common installation paths
    for path in common_paths:
        ffmpeg_exe = os.path.join(path, "ffmpeg.exe")
        if os.path.isfile(ffmpeg_exe):
            # Add to PATH if not already there
            current_path = os.environ.get('PATH', '')
            if path not in current_path:
                os.environ['PATH'] = path + os.pathsep + current_path
                print(f"✅ Added FFmpeg path to environment: {path}")
                return True
            else:
                print(f"✅ FFmpeg path already in environment: {path}")
                return True
    
    # If we get here, FFmpeg wasn't found
    print("⚠️  FFmpeg not found in common locations. Please ensure FFmpeg is installed and accessible.")
    print("   You can download FFmpeg from: https://ffmpeg.org/download.html")
    return False


if __name__ == "__main__":
    setup_ffmpeg_path()
