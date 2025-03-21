#!/usr/bin/env python
"""
inference.py

This script loads a trained FACT model checkpoint (from a directory of checkpoints),
runs auto-regressive inference to generate a long 3D dance sequence (outputting 219D frames),
converts each frame to a SMPL compatible 75D representation (3 global translation and 72 axis angle parameters),
and saves the result in a pickle file. This pickle file is compatible with the SMPL-to-FBX add-on.
"""

import os
import argparse
import pickle
import numpy as np
import tensorflow as tf
from scipy.spatial.transform import Rotation as R

# ---------------------------------------------------------------------------------------
# GPU Memory Configuration (Optional)
# ---------------------------------------------------------------------------------------
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=2000)]
        )
        print("GPU memory limit set to 2 GB.")
    except RuntimeError as e:
        print(e)

# ---------------------------------------------------------------------------------------
# Helper: Get a unique filename if the file already exists
# ---------------------------------------------------------------------------------------
def get_unique_filename(base_path):
    """
    Given a base file path, if the file exists, return a new path
    by appending an underscore and a number before the extension.
    For example, if "generated_motion.pkl" exists, it returns
    "generated_motion_1.pkl", then "generated_motion_2.pkl", etc.
    """
    if not os.path.exists(base_path):
        return base_path
    base, ext = os.path.splitext(base_path)
    i = 1
    new_path = f"{base}_{i}{ext}"
    while os.path.exists(new_path):
        i += 1
        new_path = f"{base}_{i}{ext}"
    return new_path

# ---------------------------------------------------------------------------------------
# FACT Model Definition (same as your training code)
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
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(x + ffn_output)

class TransformerDecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.mha1 = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.mha2 = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
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
        ffn_output = self.ffn(out2)
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
        self.encoder_layers = [TransformerEncoderLayer(d_model, num_heads, dff, dropout_rate)
                               for _ in range(num_encoder_layers)]
        # Motion decoder pipeline
        self.motion_proj = tf.keras.layers.Dense(d_model)
        self.motion_pos_embedding = PositionalEncoding(d_model)
        self.decoder_layers = [TransformerDecoderLayer(d_model, num_heads, dff, dropout_rate)
                               for _ in range(num_decoder_layers)]
        self.final_layer = tf.keras.layers.Dense(motion_dim)
    def call(self, inputs, training=False, enc_padding_mask=None, look_ahead_mask=None, dec_padding_mask=None):
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
            motion_features = layer(motion_features, audio_features, training,
                                    look_ahead_mask=look_ahead_mask, padding_mask=dec_padding_mask)
        return self.final_layer(motion_features)

# ---------------------------------------------------------------------------------------
# Define auto-regressive inference function
# ---------------------------------------------------------------------------------------
def infer_auto_regressive(model, inputs, steps=1200):
    """
    Runs auto-regressive inference using the model.
    Starting from a seed motion, it iteratively generates one frame at a time.
    Returns a tensor of shape (BATCH_SIZE, steps, motion_dim).
    """
    motion_input = inputs["motion_input"]
    generated_frames = []
    for i in range(steps):
        current_audio_input = inputs["audio_input"]
        output = model({"audio_input": current_audio_input, "motion_input": motion_input}, training=False)
        # Take the first frame of the output as the new frame
        new_frame = output[:, 0:1, :]
        generated_frames.append(new_frame)
        # Update seed motion: remove first frame, append new frame
        motion_input = tf.concat([motion_input[:, 1:, :], new_frame], axis=1)
    generated = tf.concat(generated_frames, axis=1)
    return generated

# ---------------------------------------------------------------------------------------
# Conversion function: Convert a 219D frame to SMPL-compatible 75D
# ---------------------------------------------------------------------------------------
def convert_219_to_75(frame_219: np.ndarray) -> np.ndarray:
    """
    Converts a single FACT output frame (219D) into a SMPL-compatible frame (75D).
    Assumes:
      - First 3 dims: global translation.
      - Next 216 dims: 24 flattened 3x3 rotation matrices.
    Returns:
      A 75D vector: 3 translation + 72 axis-angle values.
    """
    if frame_219.shape[0] != 219:
        raise ValueError(f"Expected frame dimension 219, got {frame_219.shape[0]}")
    trans = frame_219[:3]
    rot_flat = frame_219[3:]
    rot_mats = rot_flat.reshape((24, 3, 3))
    axis_angles = R.from_matrix(rot_mats).as_rotvec().reshape(-1)
    return np.concatenate([trans, axis_angles], axis=0)

# ---------------------------------------------------------------------------------------
# Main Inference Script
# ---------------------------------------------------------------------------------------
def main(args):
    # Set dimensions (must match training)
    AUDIO_DIM = 35
    MOTION_DIM = 219
    BATCH_SIZE = 1

    # Set input lengths: seed motion length and audio window length.
    seed_length = 120  # frames of seed motion
    audio_seq_length = 240  # frames of audio input for each inference step
    steps = args.steps  # total frames to generate

    # Prepare dummy inputs (or load from file if provided)
    audio_input = np.zeros((BATCH_SIZE, audio_seq_length, AUDIO_DIM), dtype=np.float32)
    seed_motion = np.zeros((BATCH_SIZE, seed_length, MOTION_DIM), dtype=np.float32)
    
    if args.audio_path is not None:
        audio_input = np.load(args.audio_path)
        print(f"Loaded audio input from {args.audio_path} with shape {audio_input.shape}")
    if args.seed_motion_path is not None:
        seed_motion = np.load(args.seed_motion_path)
        print(f"Loaded seed motion from {args.seed_motion_path} with shape {seed_motion.shape}")
    
    inputs = {"audio_input": audio_input, "motion_input": seed_motion}
    
    # Instantiate the FACT model
    model = FACTModel(
        audio_dim=AUDIO_DIM,
        motion_dim=MOTION_DIM,
        d_model=256,
        num_heads=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dff=512,
        dropout_rate=0.1
    )
    
    # Build the model by performing a dummy forward pass (ensures variables are created)
    dummy_audio = tf.zeros((BATCH_SIZE, audio_seq_length, AUDIO_DIM))
    dummy_motion = tf.zeros((BATCH_SIZE, seed_length, MOTION_DIM))
    _ = model({"audio_input": dummy_audio, "motion_input": dummy_motion}, training=False)
    
    # Load weights from checkpoint (without by_name and skip_mismatch for subclassed models)
    latest_ckpt = tf.train.latest_checkpoint(args.checkpoint_dir)
    if latest_ckpt is None:
        raise ValueError("No checkpoint found in " + args.checkpoint_dir)
    model.load_weights(latest_ckpt)
    print("Loaded checkpoint:", latest_ckpt)
    status = model.load_weights(latest_ckpt)
    status.expect_partial()  # This tells TensorFlow that it's okay if some variables are missing.
    print("Loaded checkpoint with partial restoration (as expected).")
        
    # Run auto-regressive inference to generate motion
    generated = infer_auto_regressive(model, inputs, steps=steps)  # shape: (BATCH_SIZE, steps, 219)
    generated_np = generated.numpy()[0]  # get first sample; shape: (steps, 219)
    print(f"Generated motion shape: {generated_np.shape}")
    
    # Convert each generated frame from 219D to SMPL-compatible 75D
    converted_frames = np.stack([convert_219_to_75(generated_np[t]) for t in range(generated_np.shape[0])], axis=0)
    
    # Prepare SMPL data dictionary
    smpl_trans = converted_frames[:, :3]
    smpl_poses = converted_frames[:, 3:]
    smpl_scaling = np.array([1.0], dtype=np.float32)
    smpl_data = {
        "smpl_trans": smpl_trans,
        "smpl_poses": smpl_poses,
        "smpl_scaling": smpl_scaling
    }
    
    # Save SMPL data to pickle file using a unique filename if needed
    output_path = get_unique_filename(args.output_pkl)
    with open(output_path, "wb") as f:
        pickle.dump(smpl_data, f)
    print(f"Saved SMPL-compatible pickle to '{output_path}'.")
    print(f"Output: {smpl_data['smpl_poses'].shape[0]} frames, each with {smpl_data['smpl_poses'].shape[1]} pose dims, and translations shape {smpl_data['smpl_trans'].shape}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script for FACT-based choreography generation.")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Directory containing trained FACT model checkpoints.")
    parser.add_argument("--output_pkl", type=str, required=True,
                        help="Output path for the SMPL-compatible pickle file.")
    parser.add_argument("--steps", type=int, default=1200,
                        help="Number of frames to generate (default: 1200).")
    parser.add_argument("--audio_path", type=str, default=None,
                        help="Optional path to a numpy file (.npy) with precomputed audio features.")
    parser.add_argument("--seed_motion_path", type=str, default=None,
                        help="Optional path to a numpy file (.npy) with a seed motion sequence.")
    args = parser.parse_args()
    main(args)
