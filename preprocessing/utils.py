import numpy as np
import librosa
from scipy.interpolate import interp1d

# Constants
SEQ_LEN = 240  # 4 seconds @ 60 FPS
AUDIO_DIM = 36
MOTION_DIM = 75

def extract_audio_features(y, sr, target_frames=SEQ_LEN):
    """Extract unified audio features for both datasets."""
    hop_length = int(sr / 60)
    
    # Extract base features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
    rms = librosa.feature.rms(y=y, hop_length=hop_length)
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)

    # Beat detection
    beat = np.zeros(mfcc.shape[1])
    _, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
    beat[beat_frames] = 1.0

    # Stack features
    features = np.vstack([mfcc, chroma, rms, zcr, beat.reshape(1, -1)]).T  # [T, D]

    # Resample to target length
    x_orig = np.linspace(0, 1, features.shape[0])
    x_target = np.linspace(0, 1, target_frames)
    aligned = np.zeros((target_frames, features.shape[1]))
    
    for i in range(features.shape[1]):
        f = interp1d(x_orig, features[:, i], kind="linear", fill_value="extrapolate")
        aligned[:, i] = f(x_target)

    # Ensure correct dimensionality
    if aligned.shape[1] < AUDIO_DIM:
        aligned = np.pad(aligned, ((0, 0), (0, AUDIO_DIM - aligned.shape[1])))
    else:
        aligned = aligned[:, :AUDIO_DIM]

    return aligned.astype(np.float32)

def preprocess_motion(motion, num_dancers=1):
    """Standardize motion data format across datasets."""
    # Ensure motion is 3D: [num_dancers, seq_len, motion_dim]
    if num_dancers == 1 and len(motion.shape) == 2:
        motion = motion[np.newaxis, ...]
    
    # Handle sequence length
    if motion.shape[1] < SEQ_LEN:
        pad = SEQ_LEN - motion.shape[1]
        motion = np.pad(motion, ((0, 0), (0, pad), (0, 0)), mode="edge")
    else:
        motion = motion[:, :SEQ_LEN, :]
    
    return motion.astype(np.float32)

def validate_motion(motion, num_dancers):
    """Validate motion data quality."""
    if len(motion.shape) != 3:
        raise ValueError(f"Invalid motion shape: {motion.shape}, expected 3D array")
    
    if motion.shape[0] != num_dancers:
        raise ValueError(f"Expected {num_dancers} dancers, got {motion.shape[0]}")
        
    if motion.shape[2] != MOTION_DIM:
        raise ValueError(f"Invalid motion dimensions: {motion.shape[2]}, expected {MOTION_DIM}")
        
    if np.any(np.isnan(motion)) or np.any(np.isinf(motion)):
        raise ValueError("Invalid motion values detected (NaN or Inf)")
    
    return True

def compute_statistics(data_list, axis=(0, 1)):
    """Compute mean and std statistics for normalization."""
    if not data_list:
        raise ValueError("Empty data list")
        
    data_stack = np.concatenate(data_list, axis=0)
    mean = np.mean(data_stack, axis=axis)
    std = np.std(data_stack, axis=axis) + 1e-6
    
    return mean, std

def normalize_data(data, mean=None, std=None):
    """Normalize data using given or computed statistics."""
    if mean is None or std is None:
        mean = np.mean(data, axis=(0, 1))
        std = np.std(data, axis=(0, 1)) + 1e-6
    
    return (data - mean) / std
