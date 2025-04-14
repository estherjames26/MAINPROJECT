"""Generate dance choreography from audio input."""
import os
import pickle
import numpy as np
import librosa
from typing import Dict, Any, Optional
import sys
import subprocess



from motion_stitcher.main.database import MotionDatabase
from motion_stitcher.main.stitcher import MotionStitcher
import motion_stitcher.main.config
from motion_stitcher.main.evaluate import evaluate_choreography

def load_audio(audio_path: str, sr: int = 22050) -> tuple:
    """Load audio from file."""
    try:
        y, sr = librosa.load(audio_path, sr=sr)
        duration = librosa.get_duration(y=y, sr=sr)
        print(f"Loaded audio: {os.path.basename(audio_path)}, duration: {duration:.2f}s")
        return y, sr, duration
    except Exception as e:
        print(f"Error loading audio: {e}")
        return None, None, 0

def extract_audio_features(audio: np.ndarray, sr: int) -> Dict[str, Any]:
    """Extract features from audio."""
    try:
        tempo, beat_frames = librosa.beat.beat_track(y=audio, sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        
        # Extract basic features
        features = {
            'tempo': float(tempo),
            'beat_frames': beat_frames,
            'beat_times': beat_times,
            'duration': len(audio) / sr
        }
        return features
    except Exception as e:
        print(f"Error extracting audio features: {e}")
        return {'tempo': 120.0, 'beat_frames': np.array([]), 'beat_times': np.array([]), 'duration': len(audio) / sr}

def generate_choreography(audio_path: str, num_dancers: int = 1, style: Optional[str] = None, 
                          duration: Optional[int] = None, visualise_blender: bool = False) -> str:
    """Generate choreography from audio input."""
    print(f"Generating {num_dancers}-dancer choreography for: {audio_path}")
    
    # Create output filename
    audio_basename = os.path.splitext(os.path.basename(audio_path))[0]
    output_filename = f"{audio_basename}_{num_dancers}d.pkl"
    
    # Create output path
    output_path = os.path.join(motion_stitcher.main.config.CHOREOGRAPHY_DIR, output_filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Initialise database
    db_path = os.path.join(motion_stitcher.main.config.DATABASE_DIR, f"{num_dancers}_dancer_db.pkl")
    database = MotionDatabase(db_path)
    
    # Load database
    if not database.load():
        print("Failed to load motion database")
        return None
    
    # Initialise motion stitcher
    stitcher = MotionStitcher(motion_stitcher.main.config, database)
    
    # Generate choreography and save
    success = stitcher.generate_choreography(
        audio_path=audio_path,
        output_path=output_path,
        num_dancers=num_dancers,
        target_duration=duration,
        style=style
    )
    
    if success:
        print(f"Choreography saved to: {output_path}")

        # 🌀 Then launch Blender if requested
        if visualise_blender:
            convert_script = r"C:\Users\kemij\Programming\MAINPROJECT\motion_stitcher\VISUALISATIONSMPLFBX\convert_stitching.py"
            print(f"Launching Blender visualisation with: {output_path}")
            subprocess.Popen([
                sys.executable,
                convert_script,
                "--pkl", output_path
            ])

        return output_path

# For CLI purposes only (inference)
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate dance choreography from audio")
    parser.add_argument("--audio", type=str, required=True, help="Path to audio file")
    parser.add_argument("--output", type=str, help="Path to save choreography")
    parser.add_argument("--dancers", type=int, default=1, help="Number of dancers (1-3)")
    parser.add_argument("--style", type=str, help="Dance style to use")
    parser.add_argument("--duration", type=int, help="Target duration in frames")
    parser.add_argument("--blender", action="store_true", help="Automatically visualise in Blender")

    args = parser.parse_args()

    # Validate number of dancers
    if args.dancers < 1 or args.dancers > 3:
        print("Number of dancers must be between 1 and 3")
        exit(1)

    output_path = args.output
    if output_path is None:
        # Create default output path
        output_name = f"{os.path.splitext(os.path.basename(args.audio))[0]}_{args.dancers}d.pkl"
        output_path = os.path.join(motion_stitcher.main.config.CHOREOGRAPHY_DIR, output_name)

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Generate choreography
    generate_choreography(
        audio_path=args.audio,
        num_dancers=args.dancers,
        style=args.style,
        duration=args.duration,
        visualise_blender=args.blender
    )
#C:python motion_stitcher\main\generate.py --audio r"C:\Users\kemij\Programming\MAINPROJECT\motion_stitcher\audio_input\mBR0.wav" --dancers 3