import numpy as np
import librosa
import pickle
import os

PKL_PATH = r"C:\Users\kemij\Programming\MAINPROJECT\motion_stitcher\output\choreography\mBR0_1d.pkl"
AUDIO_PATH = r"C:\Users\kemij\Programming\MAINPROJECT\motion_stitcher\audio_input\mBR0.wav" 

def evaluate_choreography(poses, trans, metadata=None):
    print("\n📊 Running evaluation...")
    if poses is None:
        print("❌ No pose data provided.")
        return

    metadata = metadata or {}
    print(f"Metadata: {metadata}")

    # Determine number of dancers
    num_dancers = poses.shape[0] if poses.ndim == 3 else 1
    num_frames = poses.shape[1] if poses.ndim == 3 else poses.shape[0]

    print(f"👯 Dancers: {num_dancers}")
    print(f"🎞️  Frames: {num_frames}")

    # Metric 1: Euclidean Smoothness
    def smoothness(motion):
        diffs = np.diff(motion, axis=0)
        dist = np.linalg.norm(diffs, axis=1)
        return np.mean(dist)

    if num_dancers == 1:
        smooth = smoothness(poses)
        print(f"📈 Smoothness (Euclidean): {smooth:.4f}")
    else:
        scores = [smoothness(poses[d]) for d in range(num_dancers)]
        print(f"📈 Avg Smoothness (Euclidean): {np.mean(scores):.4f}")

    # Metric 2: Motion Range
    pose_range = np.max(poses) - np.min(poses)
    print(f"📐 Pose Range: {pose_range:.4f}")

    # Metric 3: Beat Alignment (simple peak vs beat distance)
    print("\n🎵 Beat Alignment Estimate:")
    duration = metadata.get("duration", num_frames / 30.0)
    beat_times = metadata.get("beat_times")

    if beat_times is None:
        beat_times = np.linspace(0, duration, num=num_frames // 30)
        print("No beat_times in metadata. Using synthetic beats.")

    for d in range(num_dancers):
        motion = poses[d] if num_dancers > 1 else poses
        # simple peak positions based on movement energy
        energy = np.linalg.norm(np.diff(motion, axis=0), axis=1)
        peak_indices = np.argpartition(energy, -10)[-10:]
        peak_times = peak_indices / 30.0  # assuming 30 fps
        alignment_error = np.mean([min(abs(p - b) for b in beat_times) for p in peak_times])
        print(f" - Dancer {d+1}: Avg alignment error: {alignment_error:.2f}s")

    # Metric 4: FID-style distance using mean/variance
    print("\n🎨 FID-style Motion Quality Estimate:")
    def extract_stats(motion):
        feats = motion.reshape(-1, motion.shape[-1])
        return np.mean(feats, axis=0), np.cov(feats.T)

    try:
        # Simulated real motion statistics (replace with real dataset features)
        real_mean = np.zeros(poses.shape[-1])
        real_cov = np.eye(poses.shape[-1])

        if num_dancers == 1:
            gen_mean, gen_cov = extract_stats(poses)
        else:
            all_feats = poses.reshape(-1, poses.shape[-1])
            gen_mean, gen_cov = np.mean(all_feats, axis=0), np.cov(all_feats.T)

        mean_diff = np.sum((real_mean - gen_mean) ** 2)
        cov_diff = np.trace(real_cov + gen_cov - 2 * np.sqrt(np.dot(real_cov, gen_cov)))
        fid_score = mean_diff + cov_diff
        print(f"FID-style distance: {fid_score:.4f}")
    except Exception as fid_err:
        print("⚠️ FID computation failed:", fid_err)



def load_audio_features(audio_path, sr=22050):
    try:
        y, sr = librosa.load(audio_path, sr=sr)
        duration = librosa.get_duration(y=y, sr=sr)
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        return {
            "tempo": tempo,
            "beat_times": beat_times,
            "duration": duration
        }
    except Exception as e:
        print("⚠️ Failed to extract audio features:", e)
        return {"beat_times": None, "duration": None}

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

    if audio_path and os.path.exists(audio_path):
        print(f"🔊 Extracting beats from: {audio_path}")
        audio_feats = load_audio_features(audio_path)
        metadata.update(audio_feats)
    else:
        print("⚠️ No valid audio path found in metadata. Skipping beat analysis.")

    evaluate_choreography(poses, trans, metadata)

if __name__ == "__main__":
    main()
