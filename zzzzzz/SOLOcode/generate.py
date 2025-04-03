import os
import pickle
import numpy as np
import tensorflow as tf
from fact_model import FACTModel

# === Autoregressive Generation Config ===
AUDIO_PATH = r"C:\\Users\\kemij\\Programming\\MAINPROJECT\\zzzzzz\\dataset\\wav_npy_35d\\mBR0.npy"
SEED_MOTION_PATH = r"C:\\Users\\kemij\\Programming\\MAINPROJECT\\zzzzzz\\dataset\\motion_75d\\gBR_sBM_cAll_d04_mBR0_ch01.npy"
CHECKPOINT_DIR = r"C:\\Users\\kemij\\Programming\\MAINPROJECT\\zzzzzz\\checkpoints4"
OUTPUT_DIR = r"C:\\Users\\kemij\\Programming\\MAINPROJECT\\zzzzzz\\output"

AUDIO_DIM = 35
MOTION_DIM = 75
SEQ_LEN = 240
SEED_FRAMES = 60

# === Load Best Checkpoint ===
checkpoints = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith("cp-") and f.endswith(".ckpt.index")]
if not checkpoints:
    raise FileNotFoundError("No checkpoint files found in directory")

latest_ckpt_file = sorted(checkpoints)[-1]
ckpt_epoch = latest_ckpt_file.split("-")[1].split(".")[0]
ckpt_path = os.path.join(CHECKPOINT_DIR, f"cp-{ckpt_epoch}.ckpt")
export_path = os.path.join(OUTPUT_DIR, f"generated_autoreg_smpl{ckpt_epoch}.pkl")

# === Load Model ===
model = FACTModel(audio_dim=AUDIO_DIM, motion_dim=MOTION_DIM, d_model=128, heads=2, dff=256, num_enc=2, num_dec=2)
model.compile(optimizer=tf.keras.optimizers.Adam(), loss=lambda y_true, y_pred: tf.reduce_mean(tf.square(y_true - y_pred)))
model.load_weights(ckpt_path)

# === Load Input ===
audio = np.load(AUDIO_PATH).astype(np.float32)[:SEQ_LEN]
motion_mean, motion_std = np.mean(audio), np.std(audio)
audio = (audio - motion_mean) / motion_std  # Normalize

seed_motion = np.load(SEED_MOTION_PATH).astype(np.float32)[:SEED_FRAMES]
motion_mean, motion_std = np.mean(seed_motion), np.std(seed_motion)
seed_motion = (seed_motion - motion_mean) / motion_std  # Normalize

# === Autoregressive Inference ===
predicted_motion = np.zeros((SEQ_LEN, MOTION_DIM), dtype=np.float32)
predicted_motion[:SEED_FRAMES] = seed_motion

for i in range(SEED_FRAMES, SEQ_LEN):
    motion_input = predicted_motion.copy()
    motion_input[i:] = 0

    audio_input_exp = np.expand_dims(audio, axis=0)
    motion_input_exp = np.expand_dims(motion_input, axis=0)

    prediction = model.predict({
        "audio_input": audio_input_exp,
        "motion_input": motion_input_exp
    })

    predicted_motion[i] = prediction[0][i]

# === Convert to SMPL-format ===
smpl_poses = predicted_motion[:, :72].reshape(-1, 24, 3)
smpl_trans = predicted_motion[:, 72:]
pkl_data = {
    "smpl_poses": smpl_poses,
    "smpl_trans": smpl_trans,
    "smpl_scaling": 1.0
}

os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(export_path, "wb") as f:
    pickle.dump(pkl_data, f)

print(f"✅ Exported autoregressive motion to: {export_path}")
