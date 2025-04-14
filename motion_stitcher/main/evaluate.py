import pickle
import librosa
import numpy as np
import os
from scipy.linalg import sqrtm

# Paths
PKL_PATH = r"C:\\Users\\kemij\\Programming\\MAINPROJECT\\motion_stitcher\\output\\choreography\\output_3d.pkl"
AUDIO_PATH = r"C:\\Users\\kemij\\Programming\\MAINPROJECT\\motion_stitcher\\audio_input\\input_audio.wav"
REAL_MOTION_DIR = r"C:\\Users\\kemij\\Programming\\MAINPROJECT\\motion_stitcher\\data\\AIST\\motions"  # or AIOZ

def load_audio_features(audio_path, sr=22050):
    try:
        y, sr = librosa.load(audio_path, sr=sr)
        duration = librosa.get_duration(y=y, sr=sr)
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        return {"tempo": tempo, "beat_times": beat_times, "duration": duration}
    except Exception as e:
        print("⚠️ Failed to extract audio features:", e)
        return {"beat_times": None, "duration": None}

def extract_stats(motion):
    feats = motion.reshape(-1, motion.shape[-1])
    return np.mean(feats, axis=0), np.cov(feats.T)

def compute_fid(real_mean, real_cov, gen_mean, gen_cov):
    covmean = sqrtm(real_cov.dot(gen_cov))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    diff = real_mean - gen_mean
    return np.sum(diff**2) + np.trace(real_cov + gen_cov - 2 * covmean)

def load_motion_features(directory, limit=30):
    """Extracts mean and cov from a sample of real motion .pkl files"""
    all_feats = []
    count = 0
    for fname in os.listdir(directory):
        if not fname.endswith(".pkl"): continue
        with open(os.path.join(directory, fname), "rb") as f:
            try:
                data = pickle.load(f)
                poses = data.get("smpl_poses")
                if poses is not None:
                    if poses.ndim == 3:
                        poses = poses.reshape(-1, poses.shape[-1])
                    all_feats.append(poses)
                    count += 1
                if count >= limit:
                    break
            except Exception:
                continue
    combined = np.vstack(all_feats)
    return extract_stats(combined)

def evaluate_choreography(poses, trans, metadata=None):
    print("\n📊 Running evaluation...")
    if poses is None:
        print("❌ No pose data provided.")
        return

    metadata = metadata or {}
    print(f"Metadata: {metadata}")
    num_dancers = poses.shape[0] if poses.ndim == 3 else 1
    num_frames = poses.shape[1] if poses.ndim == 3 else poses.shape[0]

    print(f"👯 Dancers: {num_dancers}")
    print(f"🎞️  Frames: {num_frames}")

    def smoothness(motion):
        diffs = np.diff(motion, axis=0)
        return np.mean(np.linalg.norm(diffs, axis=1))

    if num_dancers == 1:
        print(f"📈 Smoothness (Euclidean): {smoothness(poses):.4f}")
    else:
        scores = [smoothness(poses[d]) for d in range(num_dancers)]
        print(f"📈 Avg Smoothness (Euclidean): {np.mean(scores):.4f}")

    print(f"📐 Pose Range: {np.max(poses) - np.min(poses):.4f}")

    print("\n🎵 Beat Alignment Estimate:")
    duration = metadata.get("duration", num_frames / 30.0)
    beat_times = metadata.get("beat_times")
    if beat_times is None:
        beat_times = np.linspace(0, duration, num=num_frames // 30)
        print("No beat_times in metadata. Using synthetic beats.")

    for d in range(num_dancers):
        motion = poses[d] if num_dancers > 1 else poses
        energy = np.linalg.norm(np.diff(motion, axis=0), axis=1)
        peaks = np.argpartition(energy, -10)[-10:]
        peak_times = peaks / 30.0
        alignment_error = np.mean([min(abs(p - b) for b in beat_times) for p in peak_times])
        print(f" - Dancer {d+1}: Avg alignment error: {alignment_error:.2f}s")

    print("\n🎨 FID with Real Motion Data:")
    try:
        gen_feats = poses.reshape(-1, poses.shape[-1]) if num_dancers > 1 else poses
        gen_mean, gen_cov = extract_stats(gen_feats)
        real_mean, real_cov = load_motion_features(REAL_MOTION_DIR)
        fid = compute_fid(real_mean, real_cov, gen_mean, gen_cov)
        print(f"✅ FID score: {fid:.4f}")
    except Exception as e:
        print("⚠️ Real FID computation failed:", e)

def main():
    if not os.path.isfile(PKL_PATH):
        print(f"❌ File not found: {PKL_PATH}")
        return

    with open(PKL_PATH, "rb") as f:
        data = pickle.load(f)

    poses = data.get("smpl_poses")
    trans = data.get("smpl_trans")
    metadata = data.get("metadata", {})

    audio_path = metadata.get("audio_path")
    if audio_path and isinstance(audio_path, str) and os.path.exists(audio_path):
        print(f"🔊 Extracting beats from: {audio_path}")
        metadata.update(load_audio_features(audio_path))
    else:
        print("⚠️ No valid audio path found in metadata. Skipping beat analysis.")

    evaluate_choreography(poses, trans, metadata)

if __name__ == "__main__":
    main()

