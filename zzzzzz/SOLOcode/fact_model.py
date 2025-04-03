import tensorflow as tf
import numpy as np

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pos = np.arange(max_len)[:, None]
        i = np.arange(d_model)[None, :]
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / d_model)
        angle_rads = pos * angle_rates
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        self.pos = tf.cast(angle_rads[None, ...], tf.float32)

    def call(self, x):
        return x + self.pos[:, :tf.shape(x)[1], :]

class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, heads, dff, dropout=0.1):
        super().__init__()
        self.att = tf.keras.layers.MultiHeadAttention(
            num_heads=heads, key_dim=d_model,
            kernel_constraint=tf.keras.constraints.MaxNorm(1.0)
        )
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model)
        ])
        self.norm1 = tf.keras.layers.LayerNormalization()
        self.norm2 = tf.keras.layers.LayerNormalization()
        self.drop1 = tf.keras.layers.Dropout(dropout)
        self.drop2 = tf.keras.layers.Dropout(dropout)

    def call(self, x, training):
        attn = self.drop1(self.att(x, x, x), training=training)
        x = self.norm1(x + attn)
        ffn = self.drop2(self.ffn(x), training=training)
        return self.norm2(x + ffn)

class TransformerDecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, heads, dff, dropout=0.1):
        super().__init__()
        self.mha1 = tf.keras.layers.MultiHeadAttention(
            num_heads=heads, key_dim=d_model,
            kernel_constraint=tf.keras.constraints.MaxNorm(1.0)
        )
        self.mha2 = tf.keras.layers.MultiHeadAttention(
            num_heads=heads, key_dim=d_model,
            kernel_constraint=tf.keras.constraints.MaxNorm(1.0)
        )
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model)
        ])
        self.norm1 = tf.keras.layers.LayerNormalization()
        self.norm2 = tf.keras.layers.LayerNormalization()
        self.norm3 = tf.keras.layers.LayerNormalization()
        self.drop1 = tf.keras.layers.Dropout(dropout)
        self.drop2 = tf.keras.layers.Dropout(dropout)
        self.drop3 = tf.keras.layers.Dropout(dropout)

    def call(self, x, enc_out, training):
        attn1 = self.drop1(self.mha1(x, x, x, use_causal_mask=True), training=training)
        x1 = self.norm1(x + attn1)
        attn2 = self.drop2(self.mha2(x1, enc_out, enc_out), training=training)
        x2 = self.norm2(x1 + attn2)
        ffn = self.drop3(self.ffn(x2), training=training)
        return self.norm3(x2 + ffn)

class FACTModel(tf.keras.Model):
    def __init__(self, audio_dim=36, motion_dim=75, d_model=128, heads=2, dff=256, num_enc=3, num_dec=3, dropout_rate=0.1):
        super().__init__()
        self.d_model = d_model
        self.audio_proj = tf.keras.layers.Dense(d_model)
        self.audio_pos = PositionalEncoding(d_model)
        self.motion_proj = tf.keras.layers.Dense(d_model)
        self.motion_pos = PositionalEncoding(d_model)
        dropout = dropout_rate

        self.encoder = [TransformerEncoderLayer(d_model, heads, dff, dropout) for _ in range(num_enc)]
        self.decoder = [TransformerDecoderLayer(d_model, heads, dff, dropout) for _ in range(num_dec)]
        self.output_layer = tf.keras.layers.Dense(motion_dim)

    def call(self, inputs, training=False):
        audio_input = inputs["audio_input"]
        motion_input = inputs["motion_input"]

        if "beat" in inputs:
            audio_input = tf.concat([audio_input, inputs["beat"]], axis=-1)

        audio_encoded = self.audio_pos(self.audio_proj(audio_input))
        for enc in self.encoder:
            audio_encoded = enc(audio_encoded, training)

        motion_encoded = self.motion_pos(self.motion_proj(motion_input))
        for dec in self.decoder:
            motion_encoded = dec(motion_encoded, audio_encoded, training)

        return self.output_layer(motion_encoded)
