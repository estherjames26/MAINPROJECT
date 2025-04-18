# classification_stitcher.py

import os
import sys
import pickle
import random
import numpy as np
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

# make sure your project root is on sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import motion_stitcher.main.config as config
from motion_stitcher.main.config import CHOREOGRAPHY_DIR, DATABASE_DIR
from motion_stitcher.main.database import MotionDatabase
from motion_stitcher.main.stitcher import MotionStitcher, compute_motion_compatibility
from motion_stitcher.main.features import MotionFeatureExtractor

# ─── Helpers to build transition dataset ─────────────────────────────────────
def load_category_sequences(db: MotionDatabase) -> List[List[str]]:
    sequences: List[List[str]] = []
    for clip_id in tqdm(db.get_all_clips(), desc="Loading clips"):
        cats = db.metadata.get(clip_id, {}).get('categories')
        if isinstance(cats, list) and len(cats) > 1:
            sequences.append(cats)
    return sequences

def build_transition_dataset(db: MotionDatabase):
    sequences = load_category_sequences(db)
    if not sequences:
        raise ValueError("No category sequences found in DB metadata.")
    cats = sorted({c for seq in sequences for c in seq})
    cat2idx = {c: i for i, c in enumerate(cats)}

    X_list, y_list = [], []
    for seq in sequences:
        for a, b in zip(seq, seq[1:]):
            xi = np.zeros(len(cats), dtype=int)
            xi[cat2idx[a]] = 1
            X_list.append(xi)
            y_list.append(cat2idx[b])

    return np.vstack(X_list), np.array(y_list, dtype=int), cat2idx

# ─── Heuristic stitcher (pure compatibility) ────────────────────────────────
class HeuristicStitcher:
    def __init__(self, db: MotionDatabase):
        extractor = MotionFeatureExtractor(window_size=60, step_size=30)
        self.windows = []
        for clip_id in db.get_all_clips():
            clip_data, _ = db.get_clip(clip_id)
            motion = clip_data.get('smpl_poses') if isinstance(clip_data, dict) else clip_data
            for win in extractor.extract_windows(motion):
                self.windows.append((clip_id, win))

    def _find_best(self, segment: np.ndarray, num_dancers: int):
        best, best_score = None, -1.0
        for _, win in self.windows:
            if num_dancers == 1:
                score = compute_motion_compatibility(segment, win)
            else:
                scores = [compute_motion_compatibility(segment[d], win[d]) for d in range(num_dancers)]
                score = sum(scores) / len(scores)
            if score > best_score:
                best, best_score = win, score
        return best

    def generate(self, audio_path: str, num_dancers: int, target_duration: Optional[int] = None):
        stitcher = MotionStitcher(config, None)  # we only use its audio extractor
        _, beat_times = stitcher._extract_audio_features(audio_path)
        choreography = self.windows[0][1].copy()
        max_frames = target_duration or 450
        for _ in beat_times:
            length = choreography.shape[0] if num_dancers==1 else choreography.shape[1]
            if length >= max_frames:
                break
            segment = choreography[-30:] if num_dancers==1 else choreography[:, -30:, :]
            next_win = self._find_best(segment, num_dancers)
            if num_dancers==1:
                choreography = np.concatenate([choreography, next_win], axis=0)
            else:
                choreography = np.concatenate([choreography, next_win], axis=1)
        return {'smpl_poses': choreography, 'metadata': {'audio_path': audio_path, 'num_dancers': num_dancers}}

# ─── Classifier‐based stitcher ───────────────────────────────────────────────
class ClassifierStitcher(MotionStitcher):
    def __init__(self, config_module, db: MotionDatabase, clf, cat2idx: Dict[str,int]):
        super().__init__(config_module, db)
        self.clf = clf
        self.cat2idx = cat2idx
        self.idx2cat = {v:k for k,v in cat2idx.items()}

    def _find_next_clip(self, choreography: np.ndarray, num_dancers: int, current_cat: str):
        xi = np.zeros(len(self.cat2idx))
        xi[self.cat2idx[current_cat]] = 1
        next_idx = self.clf.predict(xi.reshape(1, -1))[0]
        next_cat = self.idx2cat[next_idx]

        pool = self.windows_by_cat.get(next_cat, [])
        if not pool:
            return None, None

        candidates = pool if len(pool)<=10 else random.sample(pool, 10)
        best = max(
            (
                (
                    compute_motion_compatibility(
                        choreography if num_dancers==1 else choreography[:, -30:, :],
                        win         if num_dancers==1 else win[:, :30, :]
                    ), cid, win
                )
                for cid, win in candidates
            ),
            key=lambda x: x[0],
            default=(None, None, None)
        )
        return best[2], best[1]


# ─── RF+SVM hybrid wrapper ───────────────────────────────────────────────────
class RF_SVM_Ensemble:
    def __init__(self, rf, svm):
        self.rf  = rf
        self.svm = svm
    def predict(self, X):
        p = (self.rf.predict_proba(X) + self.svm.predict_proba(X)) / 2
        return np.argmax(p, axis=1)
    def predict_proba(self, X):
        return (self.rf.predict_proba(X) + self.svm.predict_proba(X)) / 2

# ─── Main: train & generate for all methods ─────────────────────────────────
def train_classifiers(db_name: str):
    db = MotionDatabase(os.path.join(DATABASE_DIR, db_name))
    if not db.load():
        raise FileNotFoundError(f"DB not found: {db_name}")

    X, y, cat2idx = build_transition_dataset(db)

    # vanilla
    rf   = RandomForestClassifier(random_state=0).fit(X, y)
    svm  = SVC(probability=True).fit(X, y)
    knn  = KNeighborsClassifier().fit(X, y)

    # tuned
    rf_grid  = GridSearchCV(RandomForestClassifier(random_state=0), {'n_estimators':[50,100,200]}, cv=2)
    rf_tuned = rf_grid.fit(X, y).best_estimator_

    svm_grid  = GridSearchCV(SVC(probability=True), {'C':[0.1,1,10]}, cv=2)
    svm_tuned = svm_grid.fit(X, y).best_estimator_

    knn_grid  = GridSearchCV(KNeighborsClassifier(), {'n_neighbors':[3,5,7]}, cv=2)
    knn_tuned = knn_grid.fit(X, y).best_estimator_

    # print params
    print("RF default params:", RandomForestClassifier(random_state=0).get_params())
    print("RF_tuned best params:", rf_tuned.get_params())
    print("SVM default params:", SVC(probability=True).get_params())
    print("SVM_tuned best params:", svm_tuned.get_params())
    print("KNN default params:", KNeighborsClassifier().get_params())
    print("KNN_tuned best params:", knn_tuned.get_params())

    # build hybrids
    hybrid_vanilla = RF_SVM_Ensemble(rf,     svm)
    hybrid_tuned   = RF_SVM_Ensemble(rf_tuned, svm_tuned)

    models = {
        'RF': rf,
        'RF_tuned': rf_tuned,
        'SVM': svm,
        'SVM_tuned': svm_tuned,
        'KNN': knn,
        'KNN_tuned': knn_tuned,
        'RF+SVM': hybrid_vanilla,
        'RF_tuned+SVM_tuned': hybrid_tuned,
    }
    return models, cat2idx

def generate_specific(test_songs_map: Dict[int, List[str]], db_name: str='1_dancer_db.pkl'):
    models, cat2idx = train_classifiers(db_name)

    for d, songs in test_songs_map.items():
        db_path = os.path.join(DATABASE_DIR, f"{d}_dancer_db.pkl")
        db = MotionDatabase(db_path)
        if not db.load():
            raise FileNotFoundError(f"DB not found: {db_path}")

        # core stitchers
        markov    = MotionStitcher(config, db)
        heuristic = HeuristicStitcher(db)

        # classifier stitchers
        cls_stitchers = {
            name: ClassifierStitcher(config, db, clf, cat2idx)
            for name, clf in models.items()
        }

        for audio in songs:
            base = os.path.splitext(os.path.basename(audio))[0]
            outputs = []

            # Markov
            outputs.append(('markov',    markov.stitch_choreography(audio, d)))
            # Heuristic
            outputs.append(('heuristic', heuristic.generate(audio, d)))
            # Each classifier & hybrid
            for name, st in cls_stitchers.items():
                outputs.append((name, st.stitch_choreography(audio, d)))

            # save them all
            for method, choreo in outputs:
                fn = f"{method}_{base}_{d}d.pkl"
                path = os.path.join(CHOREOGRAPHY_DIR, fn)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, 'wb') as f:
                    pickle.dump(choreo, f)
                print(f"Saved {method} → {path}")

if __name__ == "__main__":
    test_songs_map = {
        1: [
            r'C:\Users\kemij\Programming\MAINPROJECT\motion_stitcher\data\AIST\wav\mBR0.wav',
            r'C:\Users\kemij\Programming\MAINPROJECT\motion_stitcher\data\AIST\wav\mBR1.wav',
            r'C:\Users\kemij\Programming\MAINPROJECT\motion_stitcher\data\AIST\wav\mBR2.wav',
            r'C:\Users\kemij\Programming\MAINPROJECT\motion_stitcher\data\AIST\wav\mBR3.wav',
            r'C:\Users\kemij\Programming\MAINPROJECT\motion_stitcher\data\AIST\wav\mBR4.wav',
        ],
        2: [
            r'C:\Users\kemij\Programming\MAINPROJECT\motion_stitcher\data\AIOZ\musics\1-vFCrO-fuM_01_9_1110.wav',
            r'C:\Users\kemij\Programming\MAINPROJECT\motion_stitcher\data\AIOZ\musics\2EhlnKTJ1vo_02_0_407.wav',
            r'C:\Users\kemij\Programming\MAINPROJECT\motion_stitcher\data\AIOZ\musics\3DnbGX7dWLQ_01_41_1500.wav',
        ],
        3: [
            r'C:\Users\kemij\Programming\MAINPROJECT\motion_stitcher\data\AIOZ\musics\-4yoUMiBwXg_01_0_960.wav',
            r'C:\Users\kemij\Programming\MAINPROJECT\motion_stitcher\data\AIOZ\musics\-FXdDRM4lC0_01_22_900.wav',
            r'C:\Users\kemij\Programming\MAINPROJECT\motion_stitcher\data\AIOZ\musics\06aOlKUkZCs_01_0_1290.wav',
        ],
    }
   
    generate_specific(test_songs_map)