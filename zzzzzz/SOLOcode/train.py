import os
import csv
import numpy as np
import tensorflow as tf
from fact_model import FACTModel
import random
from collections import defaultdict

# === Config ===
AUDIO_DIM = 36  # added beat vector
MOTION_DIM = 75
SEQ_LEN = 240
SEED_FRAMES = 60
BATCH_SIZE = 24
EPOCHS = 600
LEARNING_RATE = 1e-4
FUTURE_N = 5
VAL_SPLIT = 0.15
TEST_SPLIT = 0.05

BEAT_DIR = r"C:\Users\kemij\Programming\MAINPROJECT\zzzzzz\SOLOcode\dataset\beat_vectors"
MOTION_DIR = r"C:\Users\kemij\Programming\MAINPROJECT\zzzzzz\SOLOcode\dataset\motion_75d"
AUDIO_DIR = r"C:\Users\kemij\Programming\MAINPROJECT\zzzzzz\SOLOcode\dataset\wav_npy_35d"
CHECKPOINT_DIR = r"C:\Users\kemij\Programming\MAINPROJECT\zzzzzz\SOLOcode\checkpointsSOLO5"
AUDIO_STATS = r"C:\Users\kemij\Programming\MAINPROJECT\zzzzzz\SOLOcode\dataset\audio_stats.npz"
MOTION_STATS = r"C:\Users\kemij\Programming\MAINPROJECT\zzzzzz\SOLOcode\dataset\motion_stats.npz"
LOG_CSV = os.path.join(CHECKPOINT_DIR, f"log_A{AUDIO_DIM}_M{MOTION_DIM}_B{BATCH_SIZE}_LR{LEARNING_RATE}.csv")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# === Load Stats ===
audio_stats = np.load(AUDIO_STATS)
audio_mean, audio_std = audio_stats['mean'], audio_stats['std']
motion_stats = np.load(MOTION_STATS)
motion_mean, motion_std = motion_stats['mean'], motion_stats['std']

# === Loss ===
def custom_loss(y_true, y_pred):
    y_true = tf.convert_to_tensor(y_true)
    y_pred = tf.convert_to_tensor(y_pred)
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    if y_true.shape[1] > 1:
        vel_true = y_true[:, 1:, :] - y_true[:, :-1, :]
        vel_pred = y_pred[:, 1:, :] - y_pred[:, :-1, :]
        vel_loss = tf.reduce_mean(tf.square(vel_true - vel_pred))
    else:
        vel_loss = 0.0
    return mse + 0.05 * vel_loss

# === File list split ===
all_files = sorted([f for f in os.listdir(MOTION_DIR) if f.endswith(".npy") and not f.endswith("_norm.npy")])
random.shuffle(all_files)
total = len(all_files)
val_count = int(total * VAL_SPLIT)
test_count = int(total * TEST_SPLIT)
train_count = total - val_count - test_count

train_files = all_files[:train_count]
val_files = all_files[train_count:train_count + val_count]
test_files = all_files[train_count + val_count:]

# === Data Generator ===
def data_gen(file_list):
    for fname in file_list:
        motion_path = os.path.join(MOTION_DIR, fname)

        if len(fname.split("_")) < 5:
            print(f"⚠️ Invalid motion filename format: {fname}")
            continue

        parts = fname.split("_")
        music_id = parts[4]
        audio_path = os.path.join(AUDIO_DIR, f"{music_id}.npy")
        beat_path = os.path.join(BEAT_DIR, f"{music_id}.npy")

        if not os.path.exists(audio_path):
            print(f"⚠️ Missing audio: {audio_path}")
            continue
        if not os.path.exists(beat_path):
            print(f"⚠️ Missing beat: {beat_path}")
            continue

        motion = np.load(motion_path).astype(np.float32)
        audio = np.load(audio_path).astype(np.float32)
        beat = np.load(beat_path).astype(np.float32)

        if motion.shape != (SEQ_LEN, MOTION_DIM):
            print(f"❌ Invalid motion shape for {fname}: {motion.shape}")
            continue
        if audio.shape != (SEQ_LEN, AUDIO_DIM - 1):
            print(f"❌ Invalid audio shape for {fname}: {audio.shape}")
            continue
        if beat.shape != (SEQ_LEN, 1):
            print(f"❌ Invalid beat shape for {fname}: {beat.shape}")
            continue

        audio = (audio - audio_mean) / audio_std
        motion = (motion - motion_mean) / motion_std
        audio_input = np.concatenate([audio, beat], axis=-1)

        yield ({
            "audio_input": audio_input,
            "motion_input": motion
        }, motion)

# === Datasets ===
train_dataset = tf.data.Dataset.from_generator(
    lambda: data_gen(train_files),
    output_signature=(
        {
            "audio_input": tf.TensorSpec([SEQ_LEN, AUDIO_DIM], tf.float32),
            "motion_input": tf.TensorSpec([SEQ_LEN, MOTION_DIM], tf.float32)
        },
        tf.TensorSpec([SEQ_LEN, MOTION_DIM], tf.float32)
    )
).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_generator(
    lambda: data_gen(val_files),
    output_signature=(
        {
            "audio_input": tf.TensorSpec([SEQ_LEN, AUDIO_DIM], tf.float32),
            "motion_input": tf.TensorSpec([SEQ_LEN, MOTION_DIM], tf.float32)
        },
        tf.TensorSpec([SEQ_LEN, MOTION_DIM], tf.float32)
    )
).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

test_dataset = tf.data.Dataset.from_generator(
    lambda: data_gen(test_files),
    output_signature=(
        {
            "audio_input": tf.TensorSpec([SEQ_LEN, AUDIO_DIM], tf.float32),
            "motion_input": tf.TensorSpec([SEQ_LEN, MOTION_DIM], tf.float32)
        },
        tf.TensorSpec([SEQ_LEN, MOTION_DIM], tf.float32)
    )
).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# === Model ===
model = FACTModel(audio_dim=AUDIO_DIM, motion_dim=MOTION_DIM)
model.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE), loss=custom_loss, run_eagerly=True)

# === Training ===
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint(os.path.join(CHECKPOINT_DIR, "cp-best.ckpt"), save_best_only=True, save_weights_only=True, monitor="val_loss"),
        tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True, monitor="val_loss"),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=10, monitor="val_loss")
    ]
)

# === Save CSV Log ===
with open(LOG_CSV, mode="w", newline="") as csvfile:
    writer = csv.writer(csvfile)

    metric_names = list(history.history.keys())
    writer.writerow(["epoch"] + metric_names + ["lr"])

    num_epochs = len(history.history[metric_names[0]])  # assumes all metrics have same length

    for epoch in range(num_epochs):
        row = [epoch + 1]
        for metric in metric_names:
            row.append(history.history[metric][epoch])
        lr = model.optimizer.lr.numpy()
        row.append(lr)
        writer.writerow(row)

# === Evaluation ===
test_loss = model.evaluate(test_dataset)
print("✅ Test loss:", test_loss)
print("✅ Training complete.")
