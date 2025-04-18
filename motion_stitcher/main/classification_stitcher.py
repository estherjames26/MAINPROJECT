import os
import sys
import numpy as np
from typing import List, Tuple, Dict

# ensure project root on sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from motion_stitcher.main.database import MotionDatabase


def load_category_sequences(db: MotionDatabase) -> List[List[str]]:
    """
    Extracts all category sequences from the database metadata.
    Each entry in metadata['categories'] should be a list of strings.
    """
    sequences: List[List[str]] = []
    for clip_id in db.get_all_clips():
        meta = db.metadata.get(clip_id, {})
        cats = meta.get('categories')
        if isinstance(cats, list) and len(cats) > 1:
            sequences.append(cats)
    return sequences


def build_transition_dataset(db: MotionDatabase) -> Tuple[np.ndarray, np.ndarray, Dict[str,int]]:
    """
    Build X, y for next-category classification:
    - X: one-hot encoding of current category
    - y: index of next category
    Returns (X, y, cat2idx).
    """
    sequences = load_category_sequences(db)
    if not sequences:
        raise ValueError("No category sequences found in DB metadata.")

    # collect unique categories
    cats = sorted({c for seq in sequences for c in seq})
    cat2idx = {c: i for i, c in enumerate(cats)}

    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    n_cat = len(cats)

    for seq in sequences:
        for a, b in zip(seq, seq[1:]):
            xi = np.zeros(n_cat, dtype=int)
            xi[cat2idx[a]] = 1
            X_list.append(xi)
            y_list.append(cat2idx[b])

    X = np.vstack(X_list)
    y = np.array(y_list, dtype=int)
    return X, y, cat2idx


__all__ = ['load_category_sequences', 'build_transition_dataset']
