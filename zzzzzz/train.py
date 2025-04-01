import os
import csv
import numpy as np
import tensorflow as tf
from fact_model import FACTModel

# --- Improved Loss Function: MSE + Velocity Loss ---
def custom_loss(y_true, y_pred):
    # Mean Squared Error between the full sequences
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    
    # Compute velocity: differences between consecutive frames along the time axis
    y_true_velocity = y_true[:, 1:, :] - y_true[:, :-1, :]
    y_pred_velocity = y_pred[:, 1:, :] - y_pred[:, :-1, :]
    
    # Velocity loss: MSE on velocity differences to encourage smooth transitions
    vel_loss = tf.reduce_mean(tf.square(y_true_velocity - y_pred_velocity))
    
    # Weight for velocity loss; adjust this hyperparameter as needed.
    lambda_vel = 0.1
    return mse_loss + lambda_vel * vel_loss

# --- CONFIG ---
MOTION_DIR = r"C:\Users\kemij\Programming\MAINPROJECT\zzzzzz\dataset\motion_219d"
AUDIO_DIR = r"C:\Users\kemij\Programming\MAINPROJECT\zzzzzz\dataset\wav_npy_35d"
CHECKPOINT_DIR = r"C:\Users\kemij\Programming\MAINPROJECT\zzzzzz\checkpoints2"
LOG_CSV = "zzzzzz/training_log2.csv"
EPOCHS = 3000
BATCH_SIZE = 4
AUDIO_DIM = 35
MOTION_DIM = 75
SEQ_LEN = 240

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# --- Utility to Get Sorted File Lists ---
def get_file_lists():
    audio_files = sorted([f for f in os.listdir(AUDIO_DIR) if f.endswith(".npy")])
    motion_files = sorted([f for f in os.listdir(MOTION_DIR) if f.endswith(".npy")])
    return audio_files, motion_files

# --- Data Generator for a Given List of Files ---
def data_generator(audio_files, motion_files):
    for a_file, m_file in zip(audio_files, motion_files):
        audio_path = os.path.join(AUDIO_DIR, a_file)
        motion_path = os.path.join(MOTION_DIR, m_file)
        audio = np.load(audio_path).astype(np.float32)
        motion = np.load(motion_path).astype(np.float32)
        if audio.shape[0] == SEQ_LEN and motion.shape[0] == SEQ_LEN:
            yield {"audio_input": audio, "motion_input": motion}, motion

def get_dataset(audio_files, motion_files):
    def _generator_wrapper():
        for x, y in data_generator(audio_files, motion_files):
            yield x, y
    ds = tf.data.Dataset.from_generator(
        _generator_wrapper,
        output_signature=(
            {
                "audio_input": tf.TensorSpec(shape=(SEQ_LEN, AUDIO_DIM), dtype=tf.float32),
                "motion_input": tf.TensorSpec(shape=(SEQ_LEN, MOTION_DIM), dtype=tf.float32)
            },
            tf.TensorSpec(shape=(SEQ_LEN, MOTION_DIM), dtype=tf.float32)
        )
    )
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

# --- Split Data into Training and Validation Sets ---
all_audio_files, all_motion_files = get_file_lists()
num_files = len(all_audio_files)
split_index = int(num_files * 0.8)
train_audio_files = all_audio_files[:split_index]
train_motion_files = all_motion_files[:split_index]
val_audio_files = all_audio_files[split_index:]
val_motion_files = all_motion_files[split_index:]

train_dataset = get_dataset(train_audio_files, train_motion_files)
val_dataset = get_dataset(val_audio_files, val_motion_files)

# --- Model Setup ---
model = FACTModel(audio_dim=AUDIO_DIM, motion_dim=MOTION_DIM)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0),
    loss=custom_loss
)

# --- Resume from Checkpoint ---
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

# --- Save Checkpoints Every 100 Epochs ---
class EpochModCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, interval):
        super().__init__()
        self.interval = interval
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.interval == 0:
            path = os.path.join(CHECKPOINT_DIR, f"cp-{epoch+1:04d}.ckpt")
            self.model.save_weights(path)
            print(f"Checkpoint saved: {path}")

callbacks = [
    EpochModCheckpoint(interval=100),
    CSVLogger()
]

# --- Begin Training with Validation ---
model.fit(
    train_dataset,
    epochs=initial_epoch + EPOCHS,
    initial_epoch=initial_epoch,
    validation_data=val_dataset,
    callbacks=callbacks
)
