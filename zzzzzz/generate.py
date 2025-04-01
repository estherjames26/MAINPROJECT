# autoregressive_generate_to_pkl.py - Predict motion from seed and export to .pkl

import os
import pickle
import numpy as np
import tensorflow as tf
from fact_model import FACTModel

# --- CONFIG ---
AUDIO_PATH = r"C:\Users\kemij\Programming\MAINPROJECT\zzzzzz\dataset\wav_npy_35d\mBR0.npy"
SEED_MOTION_PATH = r"C:\Users\kemij\Programming\MAINPROJECT\zzzzzz\dataset\motion_219d\gBR_sBM_cAll_d04_mBR0_ch01.npy"
CHECKPOINT = r"C:\Users\kemij\Programming\MAINPROJECT\zzzzzz\checkpoints\cp-0150.ckpt"
EXPORT_PATH = r"C:\Users\kemij\Programming\MAINPROJECT\zzzzzz\output\generated_autoreg_smpl150.pkl"
AUDIO_DIM = 35
MOTION_DIM = 75
SEQ_LEN = 240
SEED_FRAMES = 60

# --- Load model ---
model = FACTModel(audio_dim=AUDIO_DIM, motion_dim=MOTION_DIM)
model.compile(optimizer=tf.keras.optimizers.Adam(), loss=lambda y_true, y_pred: tf.reduce_mean(tf.square(y_true - y_pred)))
model.load_weights(CHECKPOINT)

# --- Load audio and seed motion ---
audio = np.load(AUDIO_PATH).astype(np.float32)[:SEQ_LEN]
seed_motion = np.load(SEED_MOTION_PATH).astype(np.float32)[:SEED_FRAMES]

# --- Prepare padded input ---
pad_len = SEQ_LEN - SEED_FRAMES
motion_input = np.concatenate([
    seed_motion,
    np.zeros((pad_len, MOTION_DIM), dtype=np.float32)
], axis=0)

# --- Predict future motion ---
audio_input = np.expand_dims(audio, axis=0)
motion_input = np.expand_dims(motion_input, axis=0)
predicted_motion = model.predict({
    "audio_input": audio_input,
    "motion_input": motion_input
})[0]

# --- Combine seed + prediction ---
full_motion = np.concatenate([
    seed_motion,
    predicted_motion[SEED_FRAMES:]
], axis=0)  # shape: [240, 75]

# --- Convert to SMPL-format .pkl ---
smpl_poses = full_motion[:, :72].reshape(-1, 24, 3)
smpl_trans = full_motion[:, 72:]
smpl_scaling = 1.0

pkl_data = {
    "smpl_poses": smpl_poses,
    "smpl_trans": smpl_trans,
    "smpl_scaling": smpl_scaling
}

with open(EXPORT_PATH, "wb") as f:
    pickle.dump(pkl_data, f)

print(f"✅ Exported autoregressive motion from seed to: {EXPORT_PATH}")
