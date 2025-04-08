"""Inference script for dance generation."""

import os
import torch
import numpy as np
from argparse import ArgumentParser
import librosa

from config import Config
from layers import GroupMotionVAE
from utils import extract_audio_features
from visualize import create_motion_animation, plot_beat_alignment

def load_model(checkpoint_path, device):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    model = GroupMotionVAE(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, config

def process_audio(audio_path, config):
    """Load and process audio file."""
    # Load audio
    y, sr = librosa.load(audio_path, sr=None)
    
    # Extract features
    features = extract_audio_features(y, sr)
    features = torch.FloatTensor(features).unsqueeze(0)  # Add batch dimension
    
    return features

@torch.no_grad()
def generate_dance(model, audio_features, num_dancers, device, temperature=1.0):
    """Generate dance motion from audio."""
    # Move audio features to device
    audio_features = audio_features.to(device)
    
    # Initialize empty motion
    B, T, _ = audio_features.shape
    motion_input = torch.zeros(
        B, T, model.config.motion_dim * num_dancers,
        device=device
    )
    
    # Generate motion
    motion_output, _, _ = model(audio_features, motion_input, num_dancers)
    
    # Add some randomness based on temperature
    if temperature > 0:
        noise = torch.randn_like(motion_output) * temperature
        motion_output = motion_output + noise
    
    return motion_output

def save_motion(motion, save_path):
    """Save generated motion to file."""
    # Convert to numpy and save
    motion_np = motion.cpu().numpy()
    np.save(save_path, motion_np)

def main():
    parser = ArgumentParser()
    parser.add_argument('--audio', required=True, help='Path to input audio file')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--output', required=True, help='Path to output directory')
    parser.add_argument('--num_dancers', type=int, default=3, help='Number of dancers')
    parser.add_argument('--temperature', type=float, default=0.1, help='Generation temperature')
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output, exist_ok=True)
    
    # Load model
    print("Loading model...")
    model, config = load_model(args.checkpoint, device)
    
    # Process audio
    print("Processing audio...")
    audio_features = process_audio(args.audio, config)
    
    # Generate dance
    print("Generating dance...")
    motion = generate_dance(
        model, audio_features, args.num_dancers,
        device, args.temperature
    )
    
    # Save outputs
    print("Saving results...")
    
    # Save motion data
    motion_path = os.path.join(args.output, 'generated_motion.npy')
    save_motion(motion, motion_path)
    
    # Create visualization
    print("Creating visualizations...")
    
    # Motion animation
    anim_path = os.path.join(args.output, 'dance_animation.gif')
    create_motion_animation(
        motion, args.num_dancers, config.motion_dim,
        save_path=anim_path
    )
    
    # Beat alignment plot
    beat_path = os.path.join(args.output, 'beat_alignment.png')
    plot_beat_alignment(motion, audio_features, save_path=beat_path)
    
    print(f"Results saved to {args.output}")

if __name__ == '__main__':
    main()
