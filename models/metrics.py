"""Evaluation metrics for dance generation."""

import numpy as np
import torch
import torch.nn.functional as F
from scipy import linalg

def compute_euclidean_distance(pred_motion, target_motion):
    """Compute average Euclidean distance between predicted and target motion."""
    # Calculate Euclidean distance for each frame and joint
    # Shape: [batch, time, joints]
    distances = torch.sqrt(torch.sum((pred_motion - target_motion) ** 2, dim=-1))
    
    # Average over all dimensions
    avg_distance = distances.mean().item()
    return avg_distance

def compute_beat_alignment(motion, audio, beat_idx=-1):
    """Compute how well motion aligns with musical beats.
    
    Args:
        motion: Motion tensor [B, T, D] or [T, D]
        audio: Audio tensor [B, T, D] or [T, D]
        beat_idx: Index of beat feature in audio tensor
        
    Returns:
        float: Beat alignment score (higher is better)
    """
    # Handle case where we don't have a valid beat index
    if beat_idx < 0 or beat_idx >= audio.shape[-1]:
        # Use the last dimension as default
        beat_idx = audio.shape[-1] - 1
    
    # Extract beat information (assume beats are binary signals)
    if audio.dim() == 3:  # [B, T, D]
        beats = audio[:, :, beat_idx] > 0.5  # [B, T]
    else:  # [T, D]
        beats = audio[:, beat_idx] > 0.5  # [T]
    
    # If no beats detected, return 0
    if beats.sum() == 0:
        return 0.0
    
    # Compute motion velocity (simple velocity based on joint positions)
    if motion.dim() == 3:  # [B, T, D]
        # For batched data
        velocity = torch.sqrt(((motion[:, 1:] - motion[:, :-1]) ** 2).sum(dim=2))  # [B, T-1]
        # Pad to match original time dimension
        velocity = torch.cat([velocity, velocity[:, -1:]], dim=1)  # [B, T]
    else:  # [T, D]
        # For single sequence
        velocity = torch.sqrt(((motion[1:] - motion[:-1]) ** 2).sum(dim=1))  # [T-1]
        # Pad to match original time dimension
        velocity = torch.cat([velocity, velocity[-1:]], dim=0)  # [T]
    
    # Normalize velocity
    if velocity.dim() == 2:  # [B, T]
        mean = velocity.mean(dim=1, keepdim=True)
        std = velocity.std(dim=1, keepdim=True) + 1e-8
        velocity_norm = (velocity - mean) / std
    else:  # [T]
        mean = velocity.mean()
        std = velocity.std() + 1e-8
        velocity_norm = (velocity - mean) / std
    
    # Compute average velocity at beat frames vs. non-beat frames
    if beats.dim() == 2:  # [B, T]
        # For each sequence in batch, compute beat alignment
        beat_scores = []
        for i in range(beats.shape[0]):
            seq_beats = beats[i]
            seq_velocity = velocity_norm[i]
            
            if seq_beats.sum() > 0 and (~seq_beats).sum() > 0:
                beat_vel = seq_velocity[seq_beats].mean()
                non_beat_vel = seq_velocity[~seq_beats].mean()
                beat_scores.append((beat_vel - non_beat_vel).item())
            else:
                beat_scores.append(0.0)
        
        return np.mean(beat_scores)
    else:  # [T]
        # Single sequence
        if beats.sum() > 0 and (~beats).sum() > 0:
            beat_vel = velocity_norm[beats].mean()
            non_beat_vel = velocity_norm[~beats].mean()
            return (beat_vel - non_beat_vel).item()
        else:
            return 0.0

def compute_motion_smoothness(motion):
    """Compute motion smoothness using acceleration."""
    accel = motion.diff(dim=1, n=2)
    smoothness = -torch.norm(accel, dim=-1).mean().item()
    return smoothness

def compute_group_synchronization(motion, num_dancers, motion_dim):
    """Compute how synchronized the dancers are."""
    B, T, _ = motion.shape
    motion = motion.view(B, T, num_dancers, motion_dim)
    
    # Compute pairwise correlations between dancers
    sync_scores = []
    for i in range(num_dancers):
        for j in range(i+1, num_dancers):
            # Compute correlation coefficient for each motion dimension
            corr = F.cosine_similarity(
                motion[..., i, :] - motion[..., i, :].mean(dim=1, keepdim=True),
                motion[..., j, :] - motion[..., j, :].mean(dim=1, keepdim=True),
                dim=1
            ).mean()
            sync_scores.append(corr.item())
    
    return np.mean(sync_scores) if sync_scores else 0.0

def compute_motion_diversity(motion_list):
    """Compute diversity of generated motions."""
    if len(motion_list) < 2:
        return 0.0
    
    # Compute pairwise distances between all motions
    distances = []
    for i in range(len(motion_list)):
        for j in range(i+1, len(motion_list)):
            dist = F.mse_loss(motion_list[i], motion_list[j]).item()
            distances.append(dist)
    
    return np.mean(distances)

def compute_style_consistency(pred_style, target_style):
    """Compute consistency between predicted and target dance styles."""
    return (pred_style == target_style).float().mean().item()

def calculate_activation_statistics(activations):
    """Calculate mean and covariance statistics for FID calculation."""
    mu = np.mean(activations, axis=0)
    sigma = np.cov(activations, rowvar=False)
    return mu, sigma

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Calculate Fréchet Distance between two multivariate Gaussians.
    
    Args:
        mu1 (np.ndarray): Mean of first distribution
        sigma1 (np.ndarray): Covariance of first distribution
        mu2 (np.ndarray): Mean of second distribution
        sigma2 (np.ndarray): Covariance of second distribution
        eps (float): Small epsilon for numerical stability
        
    Returns:
        float: Fréchet Distance
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)

def compute_fid(pred_motions, target_motions):
    """Compute Fréchet Inception Distance between predicted and target motions.
    
    Args:
        pred_motions (torch.Tensor): Predicted motions [B, T, D]
        target_motions (torch.Tensor): Target motions [B, T, D]
        
    Returns:
        float: FID score (lower is better)
    """
    # Convert to numpy arrays
    pred_np = pred_motions.reshape(pred_motions.shape[0], -1).cpu().numpy()
    target_np = target_motions.reshape(target_motions.shape[0], -1).cpu().numpy()
    
    # Calculate statistics
    mu1, sigma1 = calculate_activation_statistics(pred_np)
    mu2, sigma2 = calculate_activation_statistics(target_np)
    
    # Calculate FID
    fid_value = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    
    return fid_value

class DanceEvaluator:
    def __init__(self, config):
        self.config = config
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.metrics = {
            'euclidean_distance': [],
            'beat_alignment': [],
            'smoothness': [],
            'synchronization': [],
            'diversity': [],
            'style_consistency': []
        }
        self.generated_motions = []
        self.target_motions = []
    
    def update(self, pred_motion, target_motion, audio, num_dancers, pred_style=None, target_style=None):
        """Update metrics with new batch."""
        # Store motions for diversity and FID computation
        self.generated_motions.append(pred_motion.detach().cpu())
        self.target_motions.append(target_motion.detach().cpu())
        
        # Compute per-batch metrics
        self.metrics['euclidean_distance'].append(
            compute_euclidean_distance(pred_motion, target_motion)
        )
        
        self.metrics['beat_alignment'].append(
            compute_beat_alignment(pred_motion, audio)
        )
        
        self.metrics['smoothness'].append(
            compute_motion_smoothness(pred_motion)
        )
        
        self.metrics['synchronization'].append(
            compute_group_synchronization(pred_motion, num_dancers, self.config.motion_dim)
        )
        
        if pred_style is not None and target_style is not None:
            self.metrics['style_consistency'].append(
                compute_style_consistency(pred_style, target_style)
            )
    
    def compute(self):
        """Compute final metrics."""
        results = {}
        
        # Compute mean for each metric
        for name, values in self.metrics.items():
            if values:  # Only compute if we have values
                results[name] = np.mean(values)
        
        # Compute diversity using all generated motions
        if len(self.generated_motions) > 1:
            results['diversity'] = compute_motion_diversity(self.generated_motions)
        
        # Compute FID if we have enough samples
        if len(self.generated_motions) > 1 and len(self.target_motions) > 1:
            pred_motions = torch.cat(self.generated_motions, dim=0)
            target_motions = torch.cat(self.target_motions, dim=0)
            results['fid'] = compute_fid(pred_motions, target_motions)
        
        return results
