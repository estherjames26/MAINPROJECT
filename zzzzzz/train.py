import os
import csv
import numpy as np
import tensorflow as tf
from fact_model import FACTModel
import random
from collections import defaultdict

# === Config ===
AUDIO_DIM = 35  # added beat vector
MOTION_DIM = 75
SEQ_LEN = 240
SEED_FRAMES = 60
BATCH_SIZE = 4
EPOCHS = 3000
LEARNING_RATE = 5e-5

BEAT_DIR = "dataset/beat_vectors"
MOTION_DIR = r"C:\Users\kemij\Programming\MAINPROJECT\zzzzzz\dataset\motion_75d"
AUDIO_DIR = r"C:\Users\kemij\Programming\MAINPROJECT\zzzzzz\dataset\wav_npy_35d"
CHECKPOINT_DIR = r"C:\Users\kemij\Programming\MAINPROJECT\zzzzzz\checkpoints4"
LOG_CSV = os.path.join(CHECKPOINT_DIR, "training_log4.csv")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# === Loss Function (MSE + Velocity) ===
def custom_loss(y_true, y_pred):
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    ytv = y_true[:, 1:, :] - y_true[:, :-1, :]
    ypv = y_pred[:, 1:, :] - y_pred[:, :-1, :]
    # Increase the velocity loss weight for smoother transitions.
    vel_loss = tf.reduce_mean(tf.square(ytv - ypv))
    return mse + 0.5 * vel_loss  # Increased from 0.1 to 0.5

# === Match Motion Files to Audio Roots ===
def build_audio_motion_pairs():
    audio_files = sorted(f for f in os.listdir(AUDIO_DIR) if f.endswith(".npy"))
    motion_files = sorted(f for f in os.listdir(MOTION_DIR) if f.endswith(".npy"))
    
    audio_map = {a.split(".")[0]: a for a in audio_files}  # mBR0.npy → mBR0
    motion_map = defaultdict(list)

    for m in motion_files:
        for audio_key in audio_map:
            if audio_key in m:
                motion_map[audio_key].append(m)
                break

    full_pairs = []
    for audio_key, motions in motion_map.items():
        if audio_key in audio_map:
            for m in motions:
                full_pairs.append((audio_map[audio_key], m))
    return full_pairs

# === Generator with Beat Vector and Curriculum Masking ===
def data_generator(pairs):
    for a_file, m_file in pairs:
        a_path = os.path.join(AUDIO_DIR, a_file)
        m_path = os.path.join(MOTION_DIR, m_file)
        b_path = os.path.join(BEAT_DIR, a_file)

        audio = np.load(a_path).astype(np.float32)
        motion = np.load(m_path).astype(np.float32)
        beat = np.load(b_path).astype(np.float32) if os.path.exists(b_path) else np.zeros((SEQ_LEN, 1), dtype=np.float32)

        if audio.shape[0] == SEQ_LEN and motion.shape[0] == SEQ_LEN:
            # Optionally, you could append beat to the audio input if needed.
            mask_point = random.randint(SEED_FRAMES + 10, SEQ_LEN - 10)
            motion_input = np.concatenate([
                motion[:mask_point],
                np.zeros((SEQ_LEN - mask_point, MOTION_DIM), dtype=np.float32)
            ], axis=0)
            yield {"audio_input": audio, "motion_input": motion_input}, motion

# === Wrap Generator in tf.data.Dataset ===
def get_dataset(pairs):
    def wrapper():
        for x, y in data_generator(pairs):
            yield x, y
    return tf.data.Dataset.from_generator(
        wrapper,
        output_signature=(
            {
                "audio_input": tf.TensorSpec([SEQ_LEN, AUDIO_DIM], tf.float32),
                "motion_input": tf.TensorSpec([SEQ_LEN, MOTION_DIM], tf.float32)
            },
            tf.TensorSpec([SEQ_LEN, MOTION_DIM], tf.float32)
        )
    ).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# === Load Paired Files and Split ===
all_pairs = build_audio_motion_pairs()
split = int(0.8 * len(all_pairs))
train_ds = get_dataset(all_pairs[:split])
val_ds = get_dataset(all_pairs[split:])

# === Model & Compile ===
model = FACTModel(audio_dim=AUDIO_DIM, motion_dim=MOTION_DIM, dropout_rate=0.2)
model.compile(
    optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
    loss=custom_loss
)

# === Checkpoint Restore ===
ckpt = tf.train.latest_checkpoint(CHECKPOINT_DIR)
start_epoch = 0
if ckpt:
    model.load_weights(ckpt)
    start_epoch = int(ckpt.split("-")[1].split(".")[0])

# === Logging ===
if not os.path.exists(LOG_CSV):
    with open(LOG_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss"])

class CSVLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        val = logs.get("val_loss", -1)
        with open(LOG_CSV, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, logs['loss'], val])

class SaveEvery(tf.keras.callbacks.Callback):
    def __init__(self, interval=100):
        self.interval = interval
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.interval == 0:
            out_path = os.path.join(CHECKPOINT_DIR, f"cp-{epoch+1:04d}.ckpt")
            self.model.save_weights(out_path)
            print(f"✅ Saved checkpoint: {out_path}")

# === Training ===
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=start_epoch + EPOCHS,
    initial_epoch=start_epoch,
    callbacks=[
        SaveEvery(25),
        CSVLogger(),
        tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=10)
    ]
)

print(" Training Complete")
