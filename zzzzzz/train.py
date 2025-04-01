import os
import csv
import numpy as np
import tensorflow as tf
from fact_model import FACTModel

# Simplified loss for 75D motion
def custom_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

# --- CONFIG ---
MOTION_DIR = r"C:\Users\kemij\Programming\MAINPROJECT\zzzzzz\dataset\motion_219d"
AUDIO_DIR = r"C:\Users\kemij\Programming\MAINPROJECT\zzzzzz\dataset\wav_npy_35d"
CHECKPOINT_DIR = r"C:\Users\kemij\Programming\MAINPROJECT\zzzzzz\checkpoints"
LOG_CSV = "zzzzzz/training_log.csv"
EPOCHS = 3000
BATCH_SIZE = 4
AUDIO_DIM = 35
MOTION_DIM = 75
SEQ_LEN = 240

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# --- Prepare dataset from .npy files ---
def data_generator():
    audio_files = sorted(os.listdir(AUDIO_DIR))
    motion_files = sorted(os.listdir(MOTION_DIR))
    for a_file, m_file in zip(audio_files, motion_files):
        audio_path = os.path.join(AUDIO_DIR, a_file)
        motion_path = os.path.join(MOTION_DIR, m_file)
        if audio_path.endswith(".npy") and motion_path.endswith(".npy"):
            audio = np.load(audio_path).astype(np.float32)
            motion = np.load(motion_path).astype(np.float32)
            if audio.shape[0] == SEQ_LEN and motion.shape[0] == SEQ_LEN:
                yield {"audio_input": audio, "motion_input": motion}, motion


def get_dataset():
    def _generator_wrapper():
        for x, y in data_generator():
            yield x, y

    ds = tf.data.Dataset.from_generator(
        _generator_wrapper,
        output_signature=(
            {
                "audio_input": tf.TensorSpec(shape=(SEQ_LEN, AUDIO_DIM), dtype=tf.float32),
                "motion_input": tf.TensorSpec(shape=(SEQ_LEN, MOTION_DIM), dtype=tf.float32),
            },
            tf.TensorSpec(shape=(SEQ_LEN, MOTION_DIM), dtype=tf.float32)
        )
    )
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

# --- Training setup ---
train_dataset = get_dataset()

model = FACTModel(audio_dim=AUDIO_DIM, motion_dim=MOTION_DIM)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0),
    loss=custom_loss
)

# --- Resume from checkpoint ---
latest_ckpt = tf.train.latest_checkpoint(CHECKPOINT_DIR)
initial_epoch = 0
if latest_ckpt:
    print("Resuming from:", latest_ckpt)
    model.load_weights(latest_ckpt)
    initial_epoch = int(latest_ckpt.split('-')[1].split('.')[0])
else:
    print("No checkpoint found. Starting from scratch.")

# --- Logging ---
if not os.path.exists(LOG_CSV):
    with open(LOG_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss"])

class CSVLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        with open(LOG_CSV, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, logs['loss']])

# --- Save every 10 epochs ---
class EpochModCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, interval):
        super().__init__()
        self.interval = interval
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.interval == 0:
            path = os.path.join(CHECKPOINT_DIR, f"cp-{epoch+1:04d}.ckpt")
            self.model.save_weights(path)
            print(f"Checkpoint saved: {path}")

# --- Train ---
callbacks = [
    EpochModCheckpoint(interval=100),
    CSVLogger()
]

model.fit(
    train_dataset,
    epochs=initial_epoch + EPOCHS,
    initial_epoch=initial_epoch,
    callbacks=callbacks
)

