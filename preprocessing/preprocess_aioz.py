import os
import pickle
import numpy as np
import pandas as pd
import librosa
from utils import (
    SEQ_LEN, AUDIO_DIM, MOTION_DIM,
    extract_audio_features, preprocess_motion,
    validate_motion, compute_statistics
)

# Configuration
BASE_DIR = r"C:\Users\kemij\Programming\MAINPROJECT\zzzzzz\GROUPcode\OGdataset"
MOTION_DIR = os.path.join(BASE_DIR, "motions_smpl")
AUDIO_DIR = os.path.join(BASE_DIR, "musics")
LABELS_DIR = os.path.join(BASE_DIR, "gdance_labels")
OUT_DIR = r"C:\Users\kemij\Programming\MAINPROJECT\preprocessed\aioz"
STATS_DIR = os.path.join(OUT_DIR, "stats")
NUM_DANCERS = 3

def load_sample_list():
    """Load the list of valid samples from CSV."""
    sample_ids = set()
    
    for split in ["train", "val", "test"]:
        csv_path = os.path.join(LABELS_DIR, f"{split}_labels.csv")
        if not os.path.exists(csv_path):
            print(f"Warning: {split} labels not found at {csv_path}")
            continue
            
        df = pd.read_csv(csv_path)
        sample_ids.update(df["id"].tolist())
    
    return list(sample_ids)

def extract_group_motion(pkl_path):
    """Extract motion data for group dance."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    
    # Get SMPL poses and root translations
    poses = data["smpl_poses"]  # Shape: [num_dancers, seq_len, 72]
    trans = data["root_trans"]  # Shape: [num_dancers, seq_len, 3]
    
    # Concatenate poses and translations for each dancer
    # Final shape: [num_dancers, seq_len, 75]
    motion = np.concatenate([poses, trans], axis=-1)
    
    num_dancers = motion.shape[0]
    if num_dancers < 2 or num_dancers > 4:
        raise ValueError(f"Expected 2-4 dancers, got {num_dancers}")
    
    # Handle different numbers of dancers:
    # - If 2 dancers: pad with zeros to make it 3
    # - If 4 dancers: keep first 3 dancers
    if num_dancers == 2:
        # Create zero padding for one dancer
        pad_shape = list(motion.shape)
        pad_shape[0] = 1  # One additional dancer
        zero_pad = np.zeros(pad_shape, dtype=motion.dtype)
        motion = np.concatenate([motion, zero_pad], axis=0)
    elif num_dancers == 4:
        # Keep only first 3 dancers
        motion = motion[:3]
    
    return motion

def process_sample(sample_id):
    """Process a single group dance sample."""
    motion_path = os.path.join(MOTION_DIR, f"{sample_id}.pkl")
    wav_path = os.path.join(AUDIO_DIR, f"{sample_id}.wav")
    
    if not os.path.exists(motion_path) or not os.path.exists(wav_path):
        print(f"Missing files for {sample_id}")
        return None, None
    
    try:
        # Process motion
        motion = extract_group_motion(motion_path)
        motion = preprocess_motion(motion, num_dancers=NUM_DANCERS)
        validate_motion(motion, num_dancers=NUM_DANCERS)
        
        # Process audio
        y, sr = librosa.load(wav_path, sr=None)
        audio = extract_audio_features(y, sr)
        
        return motion, audio
        
    except Exception as e:
        print(f"Error processing {sample_id}: {e}")
        return None, None

def main():
    """Main preprocessing pipeline."""
    print("Starting AIOZ_GDANCE preprocessing...")
    
    # Create output directories
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(STATS_DIR, exist_ok=True)
    
    # Load sample list
    try:
        sample_ids = load_sample_list()
    except Exception as e:
        print(f"Error loading sample list: {e}")
        return
    
    # Process all samples
    all_motions, all_audios = [], []
    
    for sample_id in sample_ids:
        motion, audio = process_sample(sample_id)
        if motion is not None and audio is not None:
            # Save processed files
            np.save(os.path.join(OUT_DIR, f"{sample_id}_motion.npy"), motion)
            np.save(os.path.join(OUT_DIR, f"{sample_id}_audio.npy"), audio)
            
            all_motions.append(motion)
            all_audios.append(audio)
            print(f"Processed: {sample_id}")
    
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
