import os
import pickle
import numpy as np
import librosa
from utils import (
    SEQ_LEN, AUDIO_DIM, MOTION_DIM,
    extract_audio_features, preprocess_motion,
    validate_motion, compute_statistics
)

# Configuration
MOTION_DIR = r"C:\Users\kemij\Programming\MAINPROJECT\aist_plusplus_final\motions"
WAV_DIR = r"C:\Users\kemij\Programming\MAINPROJECT\wav"
OUT_DIR = r"C:\Users\kemij\Programming\MAINPROJECT\preprocessed\aist"
STATS_DIR = os.path.join(OUT_DIR, "stats")
IGNORE_LIST_FILE = os.path.join(MOTION_DIR, "ignore_list.txt")

def load_ignore_list():
    """Load the list of files to ignore."""
    if not os.path.exists(IGNORE_LIST_FILE):
        print("Ignore list not found, processing all files")
        return set()
    
    with open(IGNORE_LIST_FILE, "r") as f:
        return {line.strip() for line in f if line.strip()}

def extract_smpl_motion(pkl_path):
    """Extract SMPL motion parameters from pickle file."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    
    poses = data.get("smpl_poses")
    trans = data.get("smpl_trans")
    
    if poses is None or trans is None:
        raise ValueError("Missing SMPL parameters")
        
    return np.concatenate([poses, trans], axis=-1)  # [T, 75]

def process_sample(motion_file, ignore_list):
    """Process a single motion-audio pair."""
    if motion_file in ignore_list:
        print(f"Skipping ignored file: {motion_file}")
        return None, None
    
    # Extract audio ID
    import re
    match = re.search(r'_(m[A-Za-z0-9]+)_', motion_file)
    if not match:
        print(f"Could not extract audio ID from: {motion_file}")
        return None, None
    
    audio_id = match.group(1)
    wav_path = os.path.join(WAV_DIR, f"{audio_id}.wav")
    
    if not os.path.exists(wav_path):
        print(f"Missing audio file: {wav_path}")
        return None, None
    
    try:
        # Process motion
        motion_path = os.path.join(MOTION_DIR, motion_file)
        motion = extract_smpl_motion(motion_path)
        motion = preprocess_motion(motion, num_dancers=1)
        validate_motion(motion, num_dancers=1)
        
        # Process audio
        y, sr = librosa.load(wav_path, sr=None)
        audio = extract_audio_features(y, sr)
        
        return motion, audio
        
    except Exception as e:
        print(f"Error processing {motion_file}: {e}")
        return None, None

def main():
    """Main preprocessing pipeline."""
    print("Starting AIST++ preprocessing...")
    
    # Create output directories
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(STATS_DIR, exist_ok=True)
    
    # Load ignore list
    ignore_list = load_ignore_list()
    
    # Process all files
    motion_files = [f for f in os.listdir(MOTION_DIR) if f.endswith(".pkl")]
    all_motions, all_audios = [], []
    
    for motion_file in motion_files:
        motion, audio = process_sample(motion_file, ignore_list)
        if motion is not None and audio is not None:
            base_name = os.path.splitext(motion_file)[0]
            
            # Save processed files
            np.save(os.path.join(OUT_DIR, f"{base_name}_motion.npy"), motion)
            np.save(os.path.join(OUT_DIR, f"{base_name}_audio.npy"), audio)
            
            all_motions.append(motion)
            all_audios.append(audio)
            print(f"Processed: {base_name}")
    
    # Compute and save statistics
    if all_motions and all_audios:
        motion_mean, motion_std = compute_statistics(all_motions)
        audio_mean, audio_std = compute_statistics(all_audios)
        
        np.savez(os.path.join(STATS_DIR, "motion_stats.npz"),
                 mean=motion_mean, std=motion_std)
        np.savez(os.path.join(STATS_DIR, "audio_stats.npz"),
                 mean=audio_mean, std=audio_std)
        print("Statistics saved to:", STATS_DIR)
    else:
        print("No valid samples processed")

if __name__ == "__main__":
    main()
