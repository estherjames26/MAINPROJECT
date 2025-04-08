"""Model layers and components."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)  # [1, max_len, d_model]
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1), :]  # Broadcasting will handle batch dimension
        return self.dropout(x)

class MusicEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Linear(config.audio_dim, config.hidden_dim)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(config.hidden_dim, config.dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.ff_size,
            dropout=config.dropout,
            batch_first=True  # Important: use batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, config.num_layers)
        
        # Project to same hidden_dim instead of separate music_embedding_dim
        self.output_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        
        # Store config for gradient checkpointing
        self.config = config
    
    def forward(self, x):
        # Project input to hidden dimension
        x = self.input_proj(x)  # [batch_size, seq_len, hidden_dim]
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoder with optional gradient checkpointing
        if hasattr(self.config, 'use_gradient_checkpointing') and self.config.use_gradient_checkpointing and self.training:
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)
                return custom_forward
            
            x = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.transformer),
                x
            )
        else:
            x = self.transformer(x)  # [batch_size, seq_len, hidden_dim]
        
        # Project to embedding dimension
        x = self.output_proj(x)  # [batch_size, seq_len, hidden_dim]
        
        return x

class MotionDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # Dancer embeddings
        self.dancer_embedding = nn.Embedding(config.max_dancers, config.motion_embedding_dim)
        
        # Input projection
        self.input_proj = nn.Linear(config.latent_dim, config.hidden_dim)  # Project from latent space
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(config.hidden_dim, config.dropout)
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.ff_size,
            dropout=config.dropout,
            batch_first=True  # Important: use batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, config.num_layers)
        
        # Output projection
        self.output_proj = nn.Linear(config.hidden_dim, config.motion_dim * config.num_dancers)  # Project to all dancers
        
        # Store config for gradient checkpointing
        self.config = config
    
    def forward(self, tgt, memory, num_dancers):
        B, T, D = tgt.shape
        
        # Convert num_dancers tensor to integer
        num_dancers = num_dancers.item() if torch.is_tensor(num_dancers) else num_dancers
        
        # Create dancer embeddings - more memory efficient implementation
        dancer_ids = torch.arange(num_dancers, device=tgt.device)
        dancer_embeddings = self.dancer_embedding(dancer_ids)  # [num_dancers, motion_embedding_dim]
        
        # Project input to hidden dimension
        tgt = self.input_proj(tgt)  # [batch_size, seq_len, hidden_dim]
        
        # Add positional encoding
        tgt = self.pos_encoder(tgt)
        
        # Memory-efficient approach: Instead of expanding dancer embeddings to a large tensor,
        # we'll handle this differently in the output projection
        
        # Transformer decoder with optional gradient checkpointing
        if hasattr(self.config, 'use_gradient_checkpointing') and self.config.use_gradient_checkpointing and self.training:
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)
                return custom_forward
            
            output = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.transformer),
                tgt, memory
            )
        else:
            output = self.transformer(tgt, memory)  # [batch_size, seq_len, hidden_dim]
        
        # Project back to motion space
        output = self.output_proj(output)  # [batch_size, seq_len, num_dancers * motion_dim]
        
        return output

class GroupMotionVAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # Music encoder
        self.music_encoder = MusicEncoder(config)
        
        # Motion encoder (for training)
        self.motion_encoder = nn.Sequential(
            nn.Linear(config.motion_dim * config.num_dancers, config.hidden_dim),  # Handle all dancers at once
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.latent_dim * 2)  # mu and logvar
        )
        
        # Motion decoder
        self.motion_decoder = MotionDecoder(config)
        
        # Store config
        self.config = config
        self.latent_dim = config.latent_dim
        self.motion_dim = config.motion_dim
        self.num_dancers = config.num_dancers
    
    def encode(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, num_dancers * motion_dim]
        """
        B, T, D = x.shape
        
        # Memory-efficient encoding - process in chunks if sequence is long
        if T > 100 and self.training:
            # Process in chunks of 100 frames to save memory
            chunk_size = 100
            num_chunks = (T + chunk_size - 1) // chunk_size
            
            mu_chunks = []
            logvar_chunks = []
            
            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, T)
                
                # Extract chunk
                x_chunk = x[:, start_idx:end_idx, :]
                chunk_B, chunk_T, chunk_D = x_chunk.shape
                
                # Reshape to [batch_size * chunk_seq_len, num_dancers * motion_dim]
                x_chunk_flat = x_chunk.reshape(-1, chunk_D)
                
                # Encode
                h_chunk = self.motion_encoder(x_chunk_flat)
                
                # Reshape back
                h_chunk = h_chunk.reshape(chunk_B, chunk_T, -1)
                
                # Split into mu and logvar
                mu_chunk, logvar_chunk = h_chunk.chunk(2, dim=-1)
                
                mu_chunks.append(mu_chunk)
                logvar_chunks.append(logvar_chunk)
            
            # Concatenate chunks
            mu = torch.cat(mu_chunks, dim=1)
            logvar = torch.cat(logvar_chunks, dim=1)
        else:
            # For shorter sequences or during evaluation, process all at once
            # Reshape to [batch_size * seq_len, num_dancers * motion_dim]
            x_flat = x.reshape(-1, D)
            
            # Encode
            h = self.motion_encoder(x_flat)
            
            # Reshape back to [batch_size, seq_len, latent_dim * 2]
            h = h.reshape(B, T, -1)
            
            # Split into mu and logvar
            mu, logvar = h.chunk(2, dim=-1)
        
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, audio, motion, num_dancers):
        # Encode music
        music_encoding = self.music_encoder(audio)  # [batch_size, seq_len, hidden_dim]
        
        # Encode motion (for training)
        mu, logvar = self.encode(motion)
        z = self.reparameterize(mu, logvar)
        
        # Decode motion
        motion_output = self.motion_decoder(z, music_encoding, num_dancers)
        
        return motion_output, mu, logvar
    
    # Add a method to clear CUDA cache
    def clear_cuda_cache(self):
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
