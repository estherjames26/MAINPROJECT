import os
import numpy as np
import tensorflow as tf
from keras.layers import BatchNormalization
import addons_training as at

# ---------------------------------------------------------------------------------------
# 1. GPU Memory Configuration (Optional)
# ---------------------------------------------------------------------------------------
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Limit TensorFlow to 2 GB of GPU RAM (adjust as needed).
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=2000)]
        )
        print("GPU memory limit set to 2 GB.")
    except RuntimeError as e:
        print(e)

# ---------------------------------------------------------------------------------------
# 2. Configuration
# ---------------------------------------------------------------------------------------
TFRECORD_DIR = r"C:\Users\kemij\Programming\MAINPROJECT\mint\data"
TRAIN_FILE_PATTERN = os.path.join(TFRECORD_DIR, "aist_tfrecord-train-*")
VAL_FILE_PATTERN = os.path.join(TFRECORD_DIR, "aist_tfrecord-testval-*")  # Add this line

BATCH_SIZE = 1      # Reduce if you get OOM
EPOCHS = 100
MAX_SEQ_LEN = 600    # Truncate or randomly crop to 600 frames

CHECKPOINT_DIR = r"C:\Users\kemij\Programming\MAINPROJECT\zzzzzz\checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------------------
# 3. TFRecord Parsing
# ---------------------------------------------------------------------------------------
def _parse_function(example_proto):
    feature_description = {
        'motion_name': tf.io.FixedLenFeature([], tf.string),
        'motion_sequence': tf.io.VarLenFeature(tf.float32),
        'motion_sequence_shape': tf.io.VarLenFeature(tf.int64),
        'audio_name': tf.io.FixedLenFeature([], tf.string),
        'audio_sequence': tf.io.VarLenFeature(tf.float32),
        'audio_sequence_shape': tf.io.VarLenFeature(tf.int64),
    }
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    
    motion_seq = tf.sparse.to_dense(parsed['motion_sequence'])
    motion_shape = tf.sparse.to_dense(parsed['motion_sequence_shape'])
    motion_seq = tf.reshape(motion_seq, motion_shape)
    
    audio_seq = tf.sparse.to_dense(parsed['audio_sequence'])
    audio_shape = tf.sparse.to_dense(parsed['audio_sequence_shape'])
    audio_seq = tf.reshape(audio_seq, audio_shape)
    
    return audio_seq, motion_seq

def random_crop_or_truncate(audio_seq, motion_seq):
    """
    Crop or truncate the audio and motion sequences to a maximum length of MAX_SEQ_LEN
    to avoid extremely large T x T attention matrices.
    """
    # Truncate or randomly crop audio
    audio_len = tf.shape(audio_seq)[0]
    if audio_len > MAX_SEQ_LEN:
        start = tf.random.uniform([], 0, audio_len - MAX_SEQ_LEN, dtype=tf.int32)
        audio_seq = audio_seq[start:start+MAX_SEQ_LEN]
    
    # Truncate or randomly crop motion
    motion_len = tf.shape(motion_seq)[0]
    if motion_len > MAX_SEQ_LEN:
        start = tf.random.uniform([], 0, motion_len - MAX_SEQ_LEN, dtype=tf.int32)
        motion_seq = motion_seq[start:start+MAX_SEQ_LEN]
    
    return audio_seq, motion_seq

# ---------------------------------------------------------------------------------------
# 4. Build Dataset
# ---------------------------------------------------------------------------------------
def pack_inputs(audio, motion):
    return {"audio_input": audio, "motion_input": motion}, motion

def build_dataset(file_pattern, batch_size, shuffle=True):
    """Build a dataset from TFRecord files."""
    files = tf.data.Dataset.list_files(file_pattern)
    
    dataset = files.interleave(
        lambda x: tf.data.TFRecordDataset(x),
        cycle_length=4, block_length=16
    )
    dataset = dataset.map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Shuffle if required (usually for training data)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=100)
    
    # Crop/truncate sequences to at most MAX_SEQ_LEN
    dataset = dataset.map(random_crop_or_truncate, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Pad to consistent dimensions
    dataset = dataset.padded_batch(
        batch_size, 
        padded_shapes=([None, 35], [None, 219])
    )
    
    # Pack as dict for the model
    dataset = dataset.map(pack_inputs)
    
    # Prefetch for performance
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


train_dataset = build_dataset(TRAIN_FILE_PATTERN, BATCH_SIZE, shuffle=True)
val_dataset = build_dataset(VAL_FILE_PATTERN, BATCH_SIZE, shuffle=False)  # No need to shuffle validation

# Inspect first batch to confirm shapes
for (inputs, targets) in train_dataset.take(1):
    print(f"Audio input shape: {inputs['audio_input'].shape}")
    print(f"Motion input shape: {inputs['motion_input'].shape}")
    print(f"Target shape: {targets.shape}")

# ---------------------------------------------------------------------------------------
# 5. FACT Model Definition
# ---------------------------------------------------------------------------------------
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pos = np.arange(max_len)[:, None]
        i = np.arange(d_model)[None, :]
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        angle_rads = pos * angle_rates
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[None, ...]  # (1, max_len, d_model)
        self.pos_encoding = tf.cast(pos_encoding, tf.float32)
    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pos_encoding[:, :seq_len, :]

class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            BatchNormalization(),  # Add this line
            tf.keras.layers.Dense(d_model)
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
    def call(self, x, training, mask=None):
        attn_output = self.mha(query=x, value=x, key=x, attention_mask=mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1, training=training)  # Add training flag
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(x + ffn_output)

class TransformerDecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.mha1 = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.mha2 = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            BatchNormalization(),  # Add this line
            tf.keras.layers.Dense(d_model)
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout3 = tf.keras.layers.Dropout(dropout_rate)
    def call(self, x, enc_output, training, look_ahead_mask=None, padding_mask=None):
        # Self-attention for motion
        attn1 = self.mha1(query=x, value=x, key=x, attention_mask=look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(x + attn1)

        # Cross-attention with encoder output
        attn2 = self.mha2(query=out1, value=enc_output, key=enc_output, attention_mask=padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(out1 + attn2)

        # FFN
        ffn_output = self.ffn(out2, training=training)  # Move this line here and add training flag
        ffn_output = self.dropout3(ffn_output, training=training)
        return self.layernorm3(out2 + ffn_output)

class FACTModel(tf.keras.Model):
    def __init__(self, audio_dim, motion_dim, d_model=256, num_heads=8,
                 num_encoder_layers=4, num_decoder_layers=4, dff=512, dropout_rate=0.1):
        super(FACTModel, self).__init__()
        self.d_model = d_model
        # Audio encoder pipeline
        self.audio_proj = tf.keras.layers.Dense(d_model)
        self.audio_pos_embedding = PositionalEncoding(d_model)
        self.encoder_layers = [
            TransformerEncoderLayer(d_model, num_heads, dff, dropout_rate)
            for _ in range(num_encoder_layers)
        ]
        # Motion decoder pipeline
        self.motion_proj = tf.keras.layers.Dense(d_model)
        self.motion_pos_embedding = PositionalEncoding(d_model)
        self.decoder_layers = [
            TransformerDecoderLayer(d_model, num_heads, dff, dropout_rate)
            for _ in range(num_decoder_layers)
        ]
        self.final_layer = tf.keras.layers.Dense(motion_dim)

    def call(self, inputs, training=False, enc_padding_mask=None,
             look_ahead_mask=None, dec_padding_mask=None):
        audio_input = inputs["audio_input"]
        motion_input = inputs["motion_input"]

        # Audio encoder
        audio_features = self.audio_proj(audio_input) * tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        audio_features = self.audio_pos_embedding(audio_features)
        for layer in self.encoder_layers:
            audio_features = layer(audio_features, training, mask=enc_padding_mask)

        # Motion decoder
        motion_features = self.motion_proj(motion_input) * tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        motion_features = self.motion_pos_embedding(motion_features)
        for layer in self.decoder_layers:
            motion_features = layer(
                motion_features, audio_features, training,
                look_ahead_mask=look_ahead_mask, padding_mask=dec_padding_mask
            )

        return self.final_layer(motion_features)

# ---------------------------------------------------------------------------------------
# 6. Instantiate and Train
# ---------------------------------------------------------------------------------------
class CustomCheckpoint(tf.keras.callbacks.ModelCheckpoint):
    def on_epoch_end(self, epoch, logs=None):
        # Only save every 50 epochs (adjust the 50 as needed)
        if (epoch + 1) % 25 == 0:
            super(CustomCheckpoint, self).on_epoch_end(epoch, logs)

class TrainingMonitor(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 10 == 0:
            print(f"\nEpoch {epoch} summary:")
            print(f"Loss: {logs['loss']:.4f}")
            if 'val_loss' in logs:
                print(f"Validation Loss: {logs['val_loss']:.4f}")
            
            # Sample prediction for monitoring
            for inputs, targets in train_dataset.take(1):
                predictions = self.model.predict(inputs)
                
                # Check for NaNs or extreme values
                has_nan = np.isnan(predictions).any()
                max_val = np.max(np.abs(predictions))
                
                if has_nan:
                    print("WARNING: NaN values detected in predictions!")
                if max_val > 10:
                    print(f"WARNING: Large values in predictions: {max_val:.2f}")
                    
                print(f"Prediction shape: {predictions.shape}, Max value: {max_val:.2f}")
                break

# Add this to your callbacks list
monitor_callback = TrainingMonitor()
# Instantiate and compile your model as usual
optimizer = tf.keras.optimizers.Adam(
    learning_rate=1e-4,
    clipnorm=1.0  # Clip gradients to prevent explosion
)

# Use the custom loss function from addons_training
model = FACTModel(audio_dim=35, motion_dim=219, d_model=256, num_heads=8,
                  num_encoder_layers=4, num_decoder_layers=4, dff=512, dropout_rate=0.1)
model.compile(
    optimizer=optimizer,
    loss=at.custom_loss  # Use the imported custom loss function
)

# Define your checkpoint callback (here we save every epoch, or you could customize)
checkpoint_path = os.path.join(CHECKPOINT_DIR, "cp-{epoch:04d}.ckpt")
checkpoint_callback = CustomCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    verbose=1
)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='./logs', update_freq='epoch')

# Check for an existing checkpoint
latest_ckpt = tf.train.latest_checkpoint(CHECKPOINT_DIR)
if latest_ckpt is not None:
    print("Resuming training from checkpoint:", latest_ckpt)
    model.load_weights(latest_ckpt)
    # Optionally extract the epoch number from the checkpoint filename if you wish:
    # For a filename like "cp-0050.ckpt", you can parse the epoch as:
    initial_epoch = int(latest_ckpt.split('-')[1].split('.')[0])
else:
    print("No checkpoint found. Training from scratch.")
    initial_epoch = 0

# Continue training for additional epochs
additional_epochs = 25 # HERE TO CHANGE HOW MANY ARE TRAINED in 1 SESSION 
total_epochs = initial_epoch + additional_epochs

# Add early stopping to prevent overfitting
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=20,  # Stop after 20 epochs of no improvement
    restore_best_weights=True,
    verbose=1
)

# Update your callbacks list
callbacks = [
    checkpoint_callback,
    tensorboard_callback,
    monitor_callback,
    early_stopping  # Add early stopping
]

# Train with validation
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=total_epochs,
    initial_epoch=initial_epoch,
    callbacks=callbacks
)
