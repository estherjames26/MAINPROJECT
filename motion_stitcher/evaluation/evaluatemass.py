import os
import pickle
import numpy as np
import pandas as pd
import librosa
from scipy.linalg import sqrtm
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from motion_stitcher.main.config import (
    CHOREOGRAPHY_DIR,
    AIST_MOTION_DIR,
    AIST_WAV_DIR,
    AIOZ_DIR
)


test_clips = {
    1: ['mBR0', 'mHO0', 'mJBO', 'mJS0', 'mKR0'],                   # AIST solo IDs
    2: ['1-vFCrO-fuM_01_9_1110','2EhlnKTJ1vo_02_0_407','3DnbGX7dWLQ_01_41_1500'],  # AIOZ 2‑d IDs
    3: ['-4yoUMiBwXg_01_0_960','-FXdDRM4lC0_01_22_900','06aOlKUkZCs_01_0_1290'],  # AIOZ 3‑d IDs
}

# Metric functions
def euclidean_distance(gt, gen):
    n = min(gt.shape[0], gen.shape[0])
    return float(np.linalg.norm(gt[:n] - gen[:n], axis=1).mean())

def beat_align_score(gen_beats, audio_path, sigma=0.04):
    if not os.path.exists(audio_path):
        return np.nan
    y, sr = librosa.load(audio_path, sr=22050)
    _, bf = librosa.beat.beat_track(y=y, sr=sr)
    audio_beats = librosa.frames_to_time(bf, sr=sr)
    if len(gen_beats)==0 or len(audio_beats)==0:
        return 0.0
    return float(np.mean([np.exp(-min((t - audio_beats)**2)/(2*sigma**2)) for t in gen_beats]))

def frechet_inception_distance(gt_feats, gen_feats):
    mu_gt, mu_g = gt_feats.mean(0), gen_feats.mean(0)
    cov_gt, cov_g = np.cov(gt_feats, rowvar=False), np.cov(gen_feats, rowvar=False)
    cov_prod = sqrtm(cov_gt.dot(cov_g))
    if np.iscomplexobj(cov_prod):
        cov_prod = cov_prod.real
    return float(np.sum((mu_gt - mu_g)**2) + np.trace(cov_gt + cov_g - 2 * cov_prod))

# Build ground‑truth lookup maps for PKL files on disk

gt_map = {1: {}, 2: {}, 3: {}}

# 1‑dancer: match any AIST PKL containing "_<clipID>_" 
import glob
for p in glob.glob(os.path.join(AIST_MOTION_DIR, '*.pkl')):
    fname = os.path.basename(p)
    for cid in test_clips[1]:
        if f"_{cid}_" in fname:
            gt_map[1][cid] = p

# 2‑ & 3‑dancer: exact match in AIOZ motions_smpl
aioz_dir = os.path.join(AIOZ_DIR, 'motions_smpl')
for d in (2,3):
    for cid in test_clips[d]:
        path = os.path.join(aioz_dir, f"{cid}.pkl")
        if os.path.exists(path):
            gt_map[d][cid] = path


# Evaluation loop

records = []
for fname in sorted(os.listdir(CHOREOGRAPHY_DIR)):
    if not fname.endswith('.pkl'):
        continue

    base = fname[:-4]
    parts = base.split('_')
    tag = parts[-1]
    if len(tag) < 2 or not tag.endswith('d'):
        continue
    d = int(tag[0])

    clip_id = None
    method  = None
    for cid in test_clips[d]:
        suffix = f"_{cid}_{tag}"
        if base.endswith(suffix):
            clip_id = cid
            method  = base[:-len(suffix)]
            break
    if clip_id is None:
        continue

    # load generated choreography
    gen = pickle.load(open(os.path.join(CHOREOGRAPHY_DIR, fname), 'rb'))
    gen_pose  = gen['smpl_poses']
    gen_beats = np.array(gen['metadata'].get('beat_times', []))

    # load ground‑truth pose
    gt_path = gt_map[d].get(clip_id)
    if not gt_path:
        print(f"⚠️  No GT for {clip_id} (d={d}), skipping {fname}")
        continue
    gt_data = pickle.load(open(gt_path, 'rb'))
    gt_pose = gt_data['smpl_poses']

    # flatten to (frames, feats)
    gt_flat  = gt_pose.reshape(-1, gt_pose.shape[-1])
    gen_flat = gen_pose.reshape(-1, gen_pose.shape[-1])

    # compute metrics
    euc = euclidean_distance(gt_flat, gen_flat)
    audio_path = (os.path.join(AIST_WAV_DIR, f"{clip_id}.wav")
                  if d==1 else
                  os.path.join(AIOZ_DIR, 'musics', f"{clip_id}.wav"))
    ba  = beat_align_score(gen_beats, audio_path)
    fid = frechet_inception_distance(gt_flat, gen_flat)

    records.append({
        'method': method,
        'dancers': d,
        'clip': clip_id,
        'euclidean': euc,
        'beat_align': ba,
        'fid': fid
    })


# Summarise & print
df = pd.DataFrame(records)

# 1) Per-clip results
print("\nPer‑clip results:")
print(df.to_string(index=False))

# 2) Average per method and dancer count
avg_by_method_and_dancer = (
    df
    .groupby(['method','dancers'])[['euclidean','beat_align','fid']]
    .mean()
    .reset_index()
    .sort_values(['method','dancers'])
)
print("\nAverage per method and dancer count:")
print(avg_by_method_and_dancer.to_string(index=False))

# 3) Pivoted averages for dancers 1, 2, 3
pivot = (
    avg_by_method_and_dancer
    .pivot(index='method', columns='dancers', values=['euclidean','beat_align','fid'])
)
print("\nPivoted averages (dancer counts 1, 2, 3):")
print(pivot)

# 4) Overall averages per method
overall = (
    df
    .groupby('method')[['euclidean','beat_align','fid']]
    .mean()
    .reset_index()
)
print("\nOverall average per method:")
print(overall.to_string(index=False))


