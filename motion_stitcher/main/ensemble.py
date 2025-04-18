import os
import pickle
import joblib
import random
import numpy as np
from typing import Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

import motion_stitcher.main.config as config
from motion_stitcher.main.config import DATABASE_DIR, CHOREOGRAPHY_DIR
from motion_stitcher.main.database import MotionDatabase
from motion_stitcher.main.stitcher import MotionStitcher, compute_motion_compatibility
from motion_stitcher.main.classification_stitcher import build_transition_dataset
from motion_stitcher.main.features import MotionFeatureExtractor

# Optional: load pre‑trained genre classifier
try:
    from motion_stitcher.main.features import AudioFeatureExtractor
    _genre_clf = joblib.load(os.path.join(config.BASE_DIR, 'models', 'genre_classifier.pkl'))
except Exception:
    _genre_clf = None


def _predict_genre(audio_path: str) -> Optional[str]:
    if _genre_clf is None:
        return None
    afe = AudioFeatureExtractor()
    feats = afe.extract_features(audio_path)
    X = np.array([[feats['tempo'], feats['rms'], feats['onset_strength'], feats['duration']]])
    return _genre_clf.predict(X)[0]


def train_hyper_ensemble(
    db_name: str,
    out_model: str,
    rf_params: dict = {'n_estimators': [100, 200, 300]},
    svm_params: dict = {'C': [0.1, 1, 10]},
    cv: int = 3
) -> None:
    """
    Train RF+SVM ensemble on a transition DB and save to `out_model`.
    """
    db_path = os.path.join(DATABASE_DIR, db_name)
    db = MotionDatabase(db_path)
    if not db.load():
        raise FileNotFoundError(f"Couldn't load DB: {db_path}")

    X, y, cat2idx = build_transition_dataset(db)

    # RF tuning
    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        {'n_estimators': rf_params['n_estimators']},
        cv=cv
    )
    rf_grid.fit(X, y)
    rf = rf_grid.best_estimator_

    # SVM tuning
    svm_grid = GridSearchCV(
        SVC(probability=True),
        {'C': svm_params['C']},
        cv=cv
    )
    svm_grid.fit(X, y)
    svm = svm_grid.best_estimator_

    ensemble = {'rf': rf, 'svm': svm, 'cat2idx': cat2idx}
    os.makedirs(os.path.dirname(out_model), exist_ok=True)
    joblib.dump(ensemble, out_model)
    print(f"Saved ensemble -> {out_model}")


class EnsembleStitcher(MotionStitcher):
    """RF+SVM fused next‑category stitcher."""
    def __init__(self, db: MotionDatabase, model_path: str):
        data = joblib.load(model_path)
        self.rf = data['rf']
        self.svm = data['svm']
        self.cat2idx = data['cat2idx']
        self.idx2cat = {v: k for k, v in self.cat2idx.items()}

        super().__init__(config, db)
        # Build windows_by_cat once
        self.windows_by_cat = self._index_windows(db)

    def _index_windows(self, db: MotionDatabase):
        extractor = MotionFeatureExtractor(window_size=60, step_size=30)
        mapping = {}
        for clip_id in db.get_all_clips():
            clip_data, _ = db.get_clip(clip_id)
            # explicit motion extraction
            if isinstance(clip_data, np.ndarray):
                motion = clip_data
            elif isinstance(clip_data, dict) and 'smpl_poses' in clip_data:
                motion = clip_data['smpl_poses']
            elif isinstance(clip_data, dict) and 'motion' in clip_data:
                motion = clip_data['motion']
            else:
                continue

            windows = extractor.extract_windows(motion)
            cats = db.metadata.get(clip_id, {}).get('categories', [])
            for win, cat in zip(windows, cats):
                mapping.setdefault(cat, []).append((clip_id, win))
        return mapping

    def _find_next_clip(self, choreography, num_dancers, current_cat):
        # one-hot encode
        n_cat = len(self.cat2idx)
        xi = np.zeros((1, n_cat)); xi[0, self.cat2idx.get(current_cat,0)] = 1
        p_rf = self.rf.predict_proba(xi)[0]
        p_svm = self.svm.predict_proba(xi)[0]
        p_fuse = (p_rf + p_svm) / 2.0

        # top 3 categories
        top3 = np.argsort(p_fuse)[-3:]
        candidates = []
        for idx in top3:
            cat = self.idx2cat[idx]
            candidates += self.windows_by_cat.get(cat, [])
        if not candidates:
            candidates = sum(self.windows_by_cat.values(), [])

        subset = random.sample(candidates, min(len(candidates), 15))
        best = max(
            (
                (
                    compute_motion_compatibility(
                        choreography if num_dancers==1 else choreography[:, -30:, :],
                        win if num_dancers==1 else win[:, :30, :]
                    ), cid, win
                ) for cid, win in subset
            ), key=lambda x: x[0], default=(None,None,None)
        )
        return best[2], best[1]


def generate_ensemble_choreography(
    audio_path: str,
    num_dancers: int,
    target_duration: Optional[int] = None,
    use_genre_filter: bool = True
) -> Optional[str]:
    """Pick and run the N‑dancer ensemble, save output .pkl."""
    if num_dancers not in (1,2,3):
        raise ValueError("num_dancers must be 1,2 or 3")

    model_file = os.path.join(config.BASE_DIR, 'models', f'ensemble_{num_dancers}.pkl')
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Missing ensemble: {model_file}")

    db_file = os.path.join(DATABASE_DIR, f"{num_dancers}_dancer_db.pkl")
    db = MotionDatabase(db_file)
    if not db.load():
        raise FileNotFoundError(f"Couldn't load DB: {db_file}")

    # optional genre-based filtering
    style = None
    if use_genre_filter and _genre_clf is not None:
        style = _predict_genre(audio_path)
        if style:
            db.clips    = {cid: db.clips[cid] for cid, m in db.metadata.items() if m.get('genre')==style}
            db.metadata = {cid: db.metadata[cid] for cid in db.clips}

    stitcher = EnsembleStitcher(db, model_file)
    base = os.path.splitext(os.path.basename(audio_path))[0]
    out_name = f"{base}_{num_dancers}d_ensemble.pkl"
    out_path = os.path.join(CHOREOGRAPHY_DIR, out_name)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    result = stitcher.stitch_choreography(audio_path, num_dancers, target_duration, style)
    if result is None:
        return None

    with open(out_path, 'wb') as f:
        pickle.dump(result, f)
    return out_path


if __name__ == "__main__":
    for d in (1,2,3):
        print(f"Training ensemble for {d}-dancer…")
        train_hyper_ensemble(
            db_name   = f"{d}_dancer_db.pkl",
            out_model = os.path.join(config.BASE_DIR, 'models', f'ensemble_{d}.pkl')
        )
