import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from pathlib import Path
import librosa

# Configuration
AIOZ_DIR = r"C:\Users\kemij\Programming\MAINPROJECT\preprocessed\aioz"
LABELS_DIR = r"C:\Users\kemij\Programming\MAINPROJECT\zzzzzz\GROUPcode\OGdataset\gdance_labels"
OUTPUT_DIR = os.path.join(AIOZ_DIR, "analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_processed_samples():
    """Get list of successfully processed samples (2-4 dancers)."""
    motion_files = [f.replace("_motion.npy", "") for f in os.listdir(AIOZ_DIR) 
                   if f.endswith("_motion.npy")]
    return set(motion_files)

def load_stats():
    """Load motion and audio statistics."""
    motion_stats = np.load(os.path.join(AIOZ_DIR, "stats", "motion_stats.npz"))
    audio_stats = np.load(os.path.join(AIOZ_DIR, "stats", "audio_stats.npz"))
    return motion_stats, audio_stats

def load_labels():
    """Load and combine all label files."""
    all_labels = []
    processed_samples = get_processed_samples()
    
    for split in ["train", "val", "test"]:
        csv_path = os.path.join(LABELS_DIR, f"{split}_labels.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            # Only keep successfully processed samples (2-4 dancers)
            df = df[df["id"].isin(processed_samples)]
            df["split"] = split
            all_labels.append(df)
    
    combined_df = pd.concat(all_labels, ignore_index=True)
    print(f"Total processed samples (2-4 dancers): {len(combined_df)}")
    return combined_df

def analyze_motion_variance():
    """Analyze and plot motion variance across dimensions."""
    motion_stats, _ = load_stats()
    motion_std = motion_stats["std"]
    
    plt.figure(figsize=(15, 5))
    plt.plot(motion_std.flatten())
    plt.title("Motion Variance Across Dimensions")
    plt.xlabel("Dimension")
    plt.ylabel("Standard Deviation")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "motion_variance.png"))
    plt.close()

def analyze_genre_style_distribution():
    """Analyze and plot distribution of music genres and dance styles."""
    labels_df = load_labels()
    
    # Plot music genre distribution
    plt.figure(figsize=(12, 6))
    sns.countplot(data=labels_df, x="music_genre")
    plt.title("Distribution of Music Genres")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "genre_distribution.png"))
    plt.close()
    
    # Plot dance style distribution
    plt.figure(figsize=(12, 6))
    sns.countplot(data=labels_df, x="dance_style")
    plt.title("Distribution of Dance Styles")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "style_distribution.png"))
    plt.close()
    
    # Create genre-style heatmap
    genre_style_counts = pd.crosstab(labels_df["music_genre"], labels_df["dance_style"])
    plt.figure(figsize=(12, 8))
    sns.heatmap(genre_style_counts, annot=True, fmt="d", cmap="YlOrRd")
    plt.title("Music Genre vs Dance Style Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "genre_style_heatmap.png"))
    plt.close()

def analyze_sequence_lengths():
    """Analyze and plot distribution of sequence lengths."""
    motion_files = [f for f in os.listdir(AIOZ_DIR) if f.endswith("_motion.npy")]
    lengths = []
    
    for f in motion_files:
        motion = np.load(os.path.join(AIOZ_DIR, f))
        lengths.append(motion.shape[1])
    
    plt.figure(figsize=(10, 5))
    plt.hist(lengths, bins=50)
    plt.title("Distribution of Sequence Lengths")
    plt.xlabel("Number of Frames")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "sequence_lengths.png"))
    plt.close()

def analyze_audio_features():
    """Analyze and plot audio feature statistics."""
    _, audio_stats = load_stats()
    audio_mean = audio_stats["mean"].reshape(-1)  # Flatten to 1D
    audio_std = audio_stats["std"].reshape(-1)    # Flatten to 1D
    
    feature_groups = {
        "MFCC": slice(0, 13),
        "Chroma": slice(13, 25),
        "RMS": slice(25, 26),
        "ZCR": slice(26, 27),
        "Beat": slice(27, 28)
    }
    
    # Plot mean and std for each feature group
    plt.figure(figsize=(15, 10))
    for i, (name, slice_idx) in enumerate(feature_groups.items(), 1):
        plt.subplot(len(feature_groups), 1, i)
        plt.plot(audio_mean[slice_idx], label="Mean")
        plt.plot(audio_std[slice_idx], label="Std")
        plt.title(f"{name} Features")
        plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "audio_features.png"))
    plt.close()

def analyze_audio_feature_distribution():
    """Analyze the distribution of processed audio features."""
    print("\nAnalyzing audio feature distributions...")
    
    # Load a sample of processed audio files (up to 100 for efficiency)
    audio_files = [f for f in os.listdir(AIOZ_DIR) if f.endswith("_audio.npy")]
    if len(audio_files) > 100:
        audio_files = np.random.choice(audio_files, 100, replace=False)
    
    feature_data = []
    for f in audio_files:
        audio = np.load(os.path.join(AIOZ_DIR, f))
        feature_data.append(audio)
    
    feature_data = np.concatenate(feature_data, axis=0)  # [N*T, D]
    
    # Define feature groups
    feature_groups = {
        "MFCC": slice(0, 13),
        "Chroma": slice(13, 25),
        "RMS Energy": slice(25, 26),
        "Zero Crossing Rate": slice(26, 27),
        "Beat": slice(27, 28)
    }
    
    # Print statistical analysis
    print("\nAudio Feature Statistics:")
    print("-" * 50)
    for name, slice_idx in feature_groups.items():
        features = feature_data[:, slice_idx]
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0)
        min_val = np.min(features, axis=0)
        max_val = np.percentile(features, 99, axis=0)  # 99th percentile to avoid outliers
        
        print(f"\n{name}:")
        if features.shape[1] > 1:
            print("     Mean  |  Std   |  Min   |  Max (99th)")
            print("    -------|--------|--------|------------")
            for i in range(features.shape[1]):
                print(f"{i+1:2d}: {mean[i]:6.3f} | {std[i]:6.3f} | {min_val[i]:6.3f} | {max_val[i]:6.3f}")
        else:
            print(f"Mean: {mean[0]:.3f}")
            print(f"Std:  {std[0]:.3f}")
            print(f"Min:  {min_val[0]:.3f}")
            print(f"Max:  {max_val[0]:.3f}")
            
        # Feature-specific analysis
        if name == "MFCC":
            # Analyze temporal dynamics
            temporal_var = np.std(features, axis=0)
            most_dynamic = np.argsort(temporal_var)[-3:][::-1]
            print("\nMost dynamic MFCC coefficients:", most_dynamic + 1)
            
        elif name == "Chroma":
            # Analyze most common pitch classes
            chroma_mean = np.mean(features, axis=0)
            top_pitches = np.argsort(chroma_mean)[-3:][::-1]
            pitch_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            print("\nMost prevalent pitch classes:")
            for idx in top_pitches:
                print(f"- {pitch_names[idx]}: {chroma_mean[idx]:.3f}")
            
        elif name == "Beat":
            # Analyze beat statistics
            beat_density = np.mean(features > 0.5)  # Assuming 0.5 threshold
            avg_beat_interval = np.mean(np.diff(np.where(features > 0.5)[0]))
            print(f"\nBeat density: {beat_density:.3f}")
            if not np.isnan(avg_beat_interval):
                print(f"Average frames between beats: {avg_beat_interval:.1f}")
    
    # Create distribution plots
    plt.figure(figsize=(15, 12))
    for i, (name, slice_idx) in enumerate(feature_groups.items(), 1):
        features = feature_data[:, slice_idx]
        
        plt.subplot(len(feature_groups), 1, i)
        if features.shape[1] > 1:  # Multiple coefficients (MFCC, Chroma)
            # Box plot
            plt.subplot(len(feature_groups), 2, i*2-1)
            plt.boxplot([features[:, j] for j in range(features.shape[1])],
                       labels=[f"{j+1}" for j in range(features.shape[1])])
            plt.title(f"{name} Coefficients Distribution")
            plt.grid(True)
            
            # Correlation heatmap
            plt.subplot(len(feature_groups), 2, i*2)
            corr = np.corrcoef(features.T)
            sns.heatmap(corr, cmap='coolwarm', center=0, 
                       xticklabels=range(1, features.shape[1]+1),
                       yticklabels=range(1, features.shape[1]+1))
            plt.title(f"{name} Coefficient Correlations")
            
        else:  # Single value features
            plt.subplot(len(feature_groups), 2, i*2-1)
            plt.hist(features.flatten(), bins=50, density=True)
            plt.title(f"{name} Distribution")
            plt.grid(True)
            
            # Add temporal pattern analysis
            plt.subplot(len(feature_groups), 2, i*2)
            plt.plot(features[:1000, 0])  # Plot first 1000 frames
            plt.title(f"{name} Temporal Pattern (First 1000 frames)")
            plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "audio_feature_analysis.png"))
    plt.close()

def compare_raw_processed_audio():
    """Compare raw audio waveforms with their processed features."""
    print("\nComparing raw and processed audio...")
    
    # Load a few samples for comparison
    audio_files = [f for f in os.listdir(AIOZ_DIR) if f.endswith("_audio.npy")]
    samples = np.random.choice(audio_files, 3, replace=False)
    
    plt.figure(figsize=(15, 12))
    for i, sample in enumerate(samples, 1):
        sample_id = sample.replace("_audio.npy", "")
        wav_path = os.path.join(os.path.dirname(LABELS_DIR), "musics", f"{sample_id}.wav")
        npy_path = os.path.join(AIOZ_DIR, sample)
        
        # Load raw audio
        y, sr = librosa.load(wav_path, sr=None)
        
        # Load processed features
        features = np.load(npy_path)
        
        # Plot comparisons
        plt.subplot(3, 2, i*2-1)
        plt.plot(y[:min(len(y), sr*4)])  # Plot first 4 seconds
        plt.title(f"Sample {i}: Raw Audio Waveform")
        plt.grid(True)
        
        plt.subplot(3, 2, i*2)
        feature_groups = {
            "MFCC": features[:, :13].mean(axis=1),
            "Chroma": features[:, 13:25].mean(axis=1),
            "RMS": features[:, 25],
            "ZCR": features[:, 26],
            "Beat": features[:, 27]
        }
        for name, feat in feature_groups.items():
            plt.plot(feat, label=name, alpha=0.7)
        plt.title(f"Sample {i}: Processed Features")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "raw_vs_processed_audio.png"))
    plt.close()

def verify_audio_motion_pairs():
    """Verify that each motion file has corresponding raw and processed audio files."""
    # Get motion files from preprocessed directory
    motion_files = {f.replace("_motion.npy", "") for f in os.listdir(AIOZ_DIR) 
                   if f.endswith("_motion.npy")}
    
    # Get processed audio files
    processed_audio = {f.replace("_audio.npy", "") for f in os.listdir(AIOZ_DIR) 
                      if f.endswith("_audio.npy")}
    
    # Get raw wav files from musics directory
    music_dir = os.path.join(os.path.dirname(LABELS_DIR), "musics")
    raw_audio = {f.replace(".wav", "") for f in os.listdir(music_dir) 
                if f.endswith(".wav")}
    
    # Find mismatches
    missing_raw_audio = motion_files - raw_audio
    missing_processed_audio = motion_files - processed_audio
    missing_motion = processed_audio - motion_files
    
    # Print results
    print("\nAudio-Motion Pair Verification:")
    print(f"Total motion files: {len(motion_files)}")
    print(f"Total processed audio files: {len(processed_audio)}")
    print(f"Total raw audio files: {len(raw_audio)}")
    
    if missing_raw_audio:
        print("\nMotion files missing raw audio:")
        for sample_id in sorted(missing_raw_audio):
            print(f"- {sample_id}")
    
    if missing_processed_audio:
        print("\nMotion files missing processed audio:")
        for sample_id in sorted(missing_processed_audio):
            print(f"- {sample_id}")
    
    if missing_motion:
        print("\nProcessed audio files missing motion:")
        for sample_id in sorted(missing_motion):
            print(f"- {sample_id}")
    
    if not missing_raw_audio and not missing_processed_audio and not missing_motion:
        print("\nAll samples have matching motion, raw audio, and processed audio files!")
    
    # Return True if all pairs match
    return len(missing_raw_audio) == 0 and len(missing_processed_audio) == 0 and len(missing_motion) == 0

def main():
    """Run all analyses."""
    print("Starting dataset analysis...")
    
    # First verify audio-motion pairs
    pairs_valid = verify_audio_motion_pairs()
    if not pairs_valid:
        print("\nWarning: Some samples have missing audio or motion files!")
    
    # Run analyses
    analyze_motion_variance()
    analyze_genre_style_distribution()
    analyze_sequence_lengths()
    analyze_audio_features()
    analyze_audio_feature_distribution()
    compare_raw_processed_audio()
    
    print(f"\nAnalysis complete. Visualizations saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
