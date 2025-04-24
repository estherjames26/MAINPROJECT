import os
import numpy as np
from typing import List, Tuple, Dict
from sklearn.cluster import KMeans

from motion_stitcher.main.database import MotionDatabase
from motion_stitcher.main.features import MotionFeatureExtractor

__all__ = ['build_transition_dataset']


def build_transition_dataset(
    db: MotionDatabase,
    n_clusters: int = 10,
    window_size: int = 60,
    step_size: int = 30
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """
    Build X, y for next-state classification via KMeans clustering of motion windows:
    - Extract fixed windows from each clip
    - Flatten and cluster into n_clusters states
    - Assign cluster labels back into db.metadata['categories']
    - Create one-hot X from current cluster, y from next cluster
    Returns (X, y, cat2idx).
    """
    # Extract windows per clip
    extractor = MotionFeatureExtractor(window_size=window_size, step_size=step_size)
    windows_by_clip: Dict[str, List[np.ndarray]] = {}
    for clip_id in db.get_all_clips():
        clip_data, _ = db.get_clip(clip_id)
        if isinstance(clip_data, dict) and 'smpl_poses' in clip_data:
            motion = clip_data['smpl_poses']
        elif isinstance(clip_data, dict) and 'motion' in clip_data:
            motion = clip_data['motion']
        elif isinstance(clip_data, np.ndarray):
            motion = clip_data
        else:
            continue
        windows = extractor.extract_windows(motion)
        if windows:
            windows_by_clip[clip_id] = windows

    # Flatten windows for clustering
    flat_windows = []
    clip_window_counts = []
    for clip_id, windows in windows_by_clip.items():
        clip_window_counts.append(len(windows))
        for w in windows:
            flat_windows.append(w.flatten())
    if not flat_windows:
        raise ValueError("No motion windows found for clustering.")
    X_feat = np.vstack(flat_windows)

    # Cluster into motion states
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X_feat)

    # Assign labels back into sequences and metadata
    sequences: List[List[str]] = []
    idx = 0
    for clip_id, count in zip(windows_by_clip.keys(), clip_window_counts):
        seq_labels = [str(l) for l in labels[idx: idx + count]]
        sequences.append(seq_labels)
        # Inject into metadata
        if clip_id not in db.metadata:
            db.metadata[clip_id] = {}
        db.metadata[clip_id]['categories'] = seq_labels
        idx += count

    # Build transition dataset
    cats = sorted({c for seq in sequences for c in seq})
    cat2idx = {c: i for i, c in enumerate(cats)}

    X_list, y_list = [], []
    for seq in sequences:
        for a, b in zip(seq, seq[1:]):
            xi = np.zeros(len(cats), dtype=int)
            xi[cat2idx[a]] = 1
            X_list.append(xi)
            y_list.append(cat2idx[b])

    X = np.vstack(X_list)
    y = np.array(y_list, dtype=int)
    return X, y, cat2idx
