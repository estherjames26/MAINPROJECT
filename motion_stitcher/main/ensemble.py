# file for stitching ensemble system
import os
import pickle
import joblib
import random
import numpy as np
from typing import Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import motion_stitcher.main.config as config
from motion_stitcher.main.config import DATABASE_DIR, CHOREOGRAPHY_DIR
from motion_stitcher.main.database import MotionDatabase
from motion_stitcher.main.stitcher import MotionStitcher, compute_motion_compatibility
from motion_stitcher.main.classification_stitcher import build_transition_dataset
from motion_stitcher.main.features import MotionFeatureExtractor

def train_hyper_ensemble(
    db_name: str, out_model: str, rf_params: dict = {'n_estimators': [100, 200, 300]}, svm_params: dict = {'C': [0.1, 1, 10]}, cv: int = 3):
    
    #Train RF+SVM+KNN ensemble on a transition DB and save to `out_model`
    
    db_path = os.path.join(DATABASE_DIR, db_name)
    db = MotionDatabase(db_path)
    if not db.load():
        raise FileNotFoundError(f"Couldn't load DB: {db_path}")

    X, y, cat2idx = build_transition_dataset(db, n_clusters=10)

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

    # KNN tuning
    knn_grid = GridSearchCV(
        KNeighborsClassifier(),
        {'n_neighbors': [3, 5, 7]},
        cv=cv
    )
    knn_grid.fit(X, y)
    knn = knn_grid.best_estimator_

    # Save ensemble
    ensemble = {'rf': rf, 'svm': svm, 'knn': knn, 'cat2idx': cat2idx}
    os.makedirs(os.path.dirname(out_model), exist_ok=True)
    joblib.dump(ensemble, out_model)
    print(f"Saved ensemble -> {out_model}")


class EnsembleStitcher(MotionStitcher):
    #RF+SVM+KNN fused next‑category stitcher
    def __init__(self, db: MotionDatabase, model_path: str):
        data = joblib.load(model_path)
        self.rf      = data['rf']
        self.svm     = data['svm']
        self.knn     = data['knn']
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
        xi = np.zeros((1, n_cat))
        xi[0, self.cat2idx.get(current_cat, 0)] = 1

        p_rf  = self.rf.predict_proba(xi)[0]
        p_svm = self.svm.predict_proba(xi)[0]
        p_knn = self.knn.predict_proba(xi)[0]
        p_fuse = (p_rf + p_svm + p_knn) / 3.0

        # top 3 categories
        top3 = np.argsort(p_fuse)[-3:]
        candidates = []
        for idx in top3:
            cat = self.idx2cat[idx]
            candidates += self.windows_by_cat.get(cat, [])
        if not candidates:
            candidates = sum(self.windows_by_cat.values(), [])

        # if too many candidates, sample 15, the randomness is to ensure diversity and keep stitching quick and responsive
        subset = random.sample(candidates, min(len(candidates), 15))
        best = max(
            (
                (
                    compute_motion_compatibility(
                        choreography if num_dancers==1 else choreography[:, -30:, :],
                        win if num_dancers==1 else win[:, :30, :]
                    ), cid, win
                ) for cid, win in subset
            ), key=lambda x: x[0], default=(None, None, None)
        )
        return best[2], best[1]


def generate_ensemble_choreography(audio_path, num_dancers, target_duration) :
    #Pick and run the N‑dancer ensemble, save output .pkl
    if num_dancers not in (1,2,3):
        raise ValueError("num_dancers must be 1,2 or 3")

    model_file = os.path.join(config.BASE_DIR, 'models', f'ensemble_{num_dancers}.pkl')
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Missing ensemble: {model_file}")

    db_file = os.path.join(DATABASE_DIR, f"{num_dancers}_dancer_db.pkl")
    db = MotionDatabase(db_file)
    if not db.load():
        raise FileNotFoundError(f"Couldn't load DB: {db_file}")



    stitcher = EnsembleStitcher(db, model_file)
    base = os.path.splitext(os.path.basename(audio_path))[0]
    out_name = f"{base}_{num_dancers}d_ensemble.pkl"
    out_path = os.path.join(CHOREOGRAPHY_DIR, out_name)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    result = stitcher.stitch_choreography(audio_path, num_dancers, target_duration)
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
