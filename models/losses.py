"""Loss functions for dance generation."""

import torch
import torch.nn as nn
import torch.nn.functional as F

def reconstruction_loss(pred, target):
    """Basic L2 reconstruction loss."""
    return F.mse_loss(pred, target)

def velocity_loss(pred, target, seq_dim=1):
    """Penalize velocity differences."""
    pred_vel = pred.diff(dim=seq_dim)
    target_vel = target.diff(dim=seq_dim)
    return F.mse_loss(pred_vel, target_vel)

def acceleration_loss(pred, target, seq_dim=1):
    """Penalize acceleration differences."""
    pred_acc = pred.diff(dim=seq_dim, n=2)
    target_acc = target.diff(dim=seq_dim, n=2)
    return F.mse_loss(pred_acc, target_acc)

def smoothness_loss(motion, seq_dim=1):
    """Encourage smooth motion by minimizing jerk (third derivative)."""
    # Calculate jerk (third derivative)
    jerk = motion.diff(dim=seq_dim, n=3)
    # Minimize the magnitude of jerk
    return torch.mean(torch.norm(jerk, dim=-1))

def beat_alignment_loss(motion, audio, beat_idx=-1, seq_dim=1):
    """Encourage motion to align with musical beats."""
    # Extract beats from audio
    if beat_idx < 0:
        beat_idx = audio.shape[-1] - 1
    
    beats = audio[..., beat_idx] > 0.5  # [B, T]
    
    if beats.sum() == 0:  # No beats detected
        return torch.tensor(0.0, device=motion.device)
    
    # Calculate motion velocity
    velocity = torch.norm(motion.diff(dim=seq_dim), dim=-1)  # [B, T-1]
    # Pad to match original dimensions
    velocity = F.pad(velocity, (0, 1), "constant", 0)  # [B, T]
    
    # Normalize velocity
    mean = velocity.mean(dim=seq_dim, keepdim=True)
    std = velocity.std(dim=seq_dim, keepdim=True) + 1e-8
    velocity_norm = (velocity - mean) / std
    
    # We want higher velocity at beat frames
    beat_vel = velocity_norm[beats].mean()
    non_beat_vel = velocity_norm[~beats].mean()
    
    # Loss is negative of alignment score (since we minimize loss)
    return -1.0 * (beat_vel - non_beat_vel)

def group_coordination_loss(pred, num_dancers, motion_dim):
    """Encourage coordinated group movement."""
    B, T, _ = pred.shape
    
    # Calculate motion dimension per dancer
    motion_dim_per_dancer = motion_dim // num_dancers
    
    # Reshape to [B, T, num_dancers, motion_dim_per_dancer]
    pred = pred.reshape(B, T, num_dancers, motion_dim_per_dancer)
    
    # Calculate velocity for each dancer
    velocity = torch.norm(pred.diff(dim=1), dim=-1)  # [B, T-1, num_dancers]
    
    # Compute pairwise velocity correlations between dancers
    sync_loss = 0.0
    count = 0
    
    for i in range(num_dancers):
        for j in range(i+1, num_dancers):
            # Get velocities for dancer i and j
            vel_i = velocity[..., i]  # [B, T-1]
            vel_j = velocity[..., j]  # [B, T-1]
            
            # Normalize velocities
            vel_i_norm = (vel_i - vel_i.mean(dim=1, keepdim=True)) / (vel_i.std(dim=1, keepdim=True) + 1e-8)
            vel_j_norm = (vel_j - vel_j.mean(dim=1, keepdim=True)) / (vel_j.std(dim=1, keepdim=True) + 1e-8)
            
            # Calculate correlation (higher is better)
            corr = (vel_i_norm * vel_j_norm).mean(dim=1)  # [B]
            
            # We want to maximize correlation, so minimize negative correlation
            sync_loss -= corr.mean()
            count += 1
    
    if count > 0:
        sync_loss /= count
    
    return sync_loss

def diversity_regularization_loss(pred_batch, seq_dim=1):
    """Encourage diversity in generated motions within a batch."""
    batch_size = pred_batch.shape[0]
    if batch_size <= 1:
        return torch.tensor(0.0, device=pred_batch.device)
    
    # Calculate pairwise similarity between samples in the batch
    similarity = 0.0
    count = 0
    
    for i in range(batch_size):
        for j in range(i+1, batch_size):
            # Calculate cosine similarity between flattened motions
            motion_i = pred_batch[i].reshape(-1)
            motion_j = pred_batch[j].reshape(-1)
            
            sim = F.cosine_similarity(motion_i.unsqueeze(0), motion_j.unsqueeze(0))
            similarity += sim
            count += 1
    
    if count > 0:
        similarity /= count
    
    # We want to minimize similarity to encourage diversity
    return similarity

def kl_loss(mu, logvar):
    """KL divergence loss for VAE."""
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

class DanceGenerationLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def forward(self, pred, target, mu, logvar, num_dancers, audio=None):
        """Compute comprehensive loss with all components."""
        # Basic reconstruction loss
        loss_recon = reconstruction_loss(pred, target)
        
        # Velocity and acceleration losses for smoother motion
        loss_vel = velocity_loss(pred, target) if self.config.w_velocity > 0 else 0.0
        loss_acc = acceleration_loss(pred, target) if self.config.w_acceleration > 0 else 0.0
        
        # Additional smoothness loss
        loss_smooth = smoothness_loss(pred) if hasattr(self.config, 'w_smoothness') and self.config.w_smoothness > 0 else 0.0
        
        # Group synchronization loss
        loss_sync = group_coordination_loss(pred, num_dancers, self.config.motion_dim) if num_dancers > 1 and self.config.w_sync > 0 else 0.0
        
        # KL divergence loss for VAE
        loss_kld = kl_loss(mu, logvar) / (pred.shape[0] * pred.shape[1]) if self.config.w_kld > 0 else 0.0
        
        # Beat alignment loss
        loss_beat = 0.0
        if audio is not None and hasattr(self.config, 'w_beat') and self.config.w_beat > 0:
            loss_beat = beat_alignment_loss(pred, audio)
        
        # Diversity regularization
        loss_div = 0.0
        if pred.shape[0] > 1 and hasattr(self.config, 'w_diversity') and self.config.w_diversity > 0:
            loss_div = diversity_regularization_loss(pred)
        
        # Combine all losses with their weights
        total_loss = (
            self.config.w_recon * loss_recon +
            self.config.w_velocity * loss_vel +
            self.config.w_acceleration * loss_acc +
            self.config.w_kld * loss_kld +
            self.config.w_sync * loss_sync
        )
        
        # Add optional losses if configured
        if hasattr(self.config, 'w_smoothness'):
            total_loss += self.config.w_smoothness * loss_smooth
        
        if hasattr(self.config, 'w_beat') and audio is not None:
            total_loss += self.config.w_beat * loss_beat
        
        if hasattr(self.config, 'w_diversity') and pred.shape[0] > 1:
            total_loss += self.config.w_diversity * loss_div
        
        # Return individual loss components for logging
        components = {
            'recon': float(loss_recon.item()),
            'vel': float(loss_vel.item()) if isinstance(loss_vel, torch.Tensor) else loss_vel,
            'acc': float(loss_acc.item()) if isinstance(loss_acc, torch.Tensor) else loss_acc,
            'kld': float(loss_kld.item()) if isinstance(loss_kld, torch.Tensor) else loss_kld,
            'sync': float(loss_sync.item()) if isinstance(loss_sync, torch.Tensor) else loss_sync,
            'total': float(total_loss.item())
        }
        
        # Add optional components if used
        if hasattr(self.config, 'w_smoothness'):
            components['smooth'] = float(loss_smooth.item()) if isinstance(loss_smooth, torch.Tensor) else loss_smooth
        
        if hasattr(self.config, 'w_beat') and audio is not None:
            components['beat'] = float(loss_beat.item()) if isinstance(loss_beat, torch.Tensor) else loss_beat
        
        if hasattr(self.config, 'w_diversity') and pred.shape[0] > 1:
            components['div'] = float(loss_div.item()) if isinstance(loss_div, torch.Tensor) else loss_div
        
        return total_loss, components
