"""Configuration settings for the motion stitching system."""
import os

# Define base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MOTION_STITCHER_DIR = os.path.join(BASE_DIR, "motion_stitcher")

# Data directories
AIST_MOTION_DIR = os.path.join(MOTION_STITCHER_DIR, "data", "AIST", "motions")
AIST_WAV_DIR = os.path.join(MOTION_STITCHER_DIR, "data", "AIST", "wav")
AIOZ_DIR = os.path.join(MOTION_STITCHER_DIR, "data", "AIOZ")

# Output directories
OUTPUT_DIR = os.path.join(MOTION_STITCHER_DIR, "output")
DATABASE_DIR = os.path.join(OUTPUT_DIR, "database")
AUDIO_FEAT_DIR = os.path.join(OUTPUT_DIR, "audio_features")
CHOREOGRAPHY_DIR = os.path.join(OUTPUT_DIR, "choreography")
VISUALISATION_DIR = os.path.join(OUTPUT_DIR, "videos")
AUDIO_INPUT_DIR = os.path.join(MOTION_STITCHER_DIR, "audio_input")

# Create directories if they don't exist
def create_directories():
    """Create necessary output directories"""
    directories = [
        OUTPUT_DIR,
        DATABASE_DIR,
        AUDIO_FEAT_DIR,
        CHOREOGRAPHY_DIR,
        VISUALISATION_DIR,
        AUDIO_INPUT_DIR
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

# Create directories when module is imported
create_directories()
