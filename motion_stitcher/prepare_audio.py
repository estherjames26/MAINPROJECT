"""Utilities for audio file preparation and copying."""
import os
import shutil
import glob
import librosa
import soundfile as sf
import numpy as np
from config import Config

def copy_audio_file(source_path, normalize=True):
    """Copy an audio file to the audio_input directory."""
    # Load config
    config = Config()
    
    if not os.path.exists(source_path):
        print(f"Error: Source audio file not found at {source_path}")
        return None
    
    # Create input directory if it doesn't exist
    audio_input_dir = config.audio_input_dir
    os.makedirs(audio_input_dir, exist_ok=True)
    
    # Get destination filename
    dest_filename = os.path.basename(source_path)
    dest_path = os.path.join(audio_input_dir, dest_filename)
    
    if normalize:
        # Load, normalize, and save the audio
        try:
            y, sr = librosa.load(source_path, sr=None)
            
            # Normalize audio
            y_norm = librosa.util.normalize(y)
            
            # Save normalized audio
            sf.write(dest_path, y_norm, sr)
            print(f"Normalized and copied audio to {dest_path}")
        except Exception as e:
            print(f"Error normalizing audio: {e}")
            print("Falling back to direct copy...")
            shutil.copyfile(source_path, dest_path)
    else:
        # Direct copy
        shutil.copyfile(source_path, dest_path)
        print(f"Copied audio to {dest_path}")
    
    return dest_path

def copy_example_audio():
    """Copy an example audio file to the audio_input directory."""
    # Load config
    config = Config()
    
    # Look for example audio in the AIST++ WAV directory
    wav_dir = config.aist_wav_dir
    
    if not os.path.exists(wav_dir):
        print(f"Error: WAV directory not found at {wav_dir}")
        return None
    
    # Find the first .wav file
    wav_files = glob.glob(os.path.join(wav_dir, "*.wav"))
    
    if not wav_files:
        print(f"Error: No WAV files found in {wav_dir}")
        return None
    
    example_file = wav_files[0]
    return copy_audio_file(example_file, normalize=True)

def trim_audio(audio_path, duration=60):
    """Trim an audio file to the specified duration in seconds."""
    # Load config
    config = Config()
    
    try:
        # Create output path
        filename = os.path.basename(audio_path)
        base_name, ext = os.path.splitext(filename)
        output_filename = f"trimmed_{base_name}{ext}"
        output_path = os.path.join(config.audio_input_dir, output_filename)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Load audio
        y, sr = librosa.load(audio_path, sr=None)
        
        # Calculate samples for desired duration
        samples = int(duration * sr)
        
        # Trim if necessary
        if len(y) > samples:
            y_trimmed = y[:samples]
        else:
            y_trimmed = y
            print(f"Warning: Audio is shorter than {duration}s, not trimming")
        
        # Save trimmed audio
        sf.write(output_path, y_trimmed, sr)
        print(f"Trimmed audio saved to {output_path}")
        
        return output_path
    except Exception as e:
        print(f"Error trimming audio: {e}")
        return None

def prepare_audio_file(source_path, normalize=True, trim_duration=0):
    """Prepare audio file by copying and optionally normalizing and trimming."""
    copied_path = copy_audio_file(source_path, normalize=normalize)
    
    if copied_path and trim_duration > 0:
        return trim_audio(copied_path, duration=trim_duration)
    
    return copied_path

# For CLI purposes only (inference and visualization)
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare audio files for motion stitching")
    parser.add_argument("--source", "-s", help="Path to source audio file")
    parser.add_argument("--example", "-e", action="store_true", help="Copy an example audio file")
    parser.add_argument("--no-normalize", action="store_true", help="Skip audio normalization")
    parser.add_argument("--trim", "-t", type=int, default=0, help="Trim audio to specified seconds")
    
    args = parser.parse_args()
    
    if args.example:
        audio_path = copy_example_audio()
    elif args.source:
        audio_path = prepare_audio_file(args.source, normalize=not args.no_normalize, trim_duration=args.trim)
    else:
        print("Please specify an audio file with --source or use --example")
        audio_path = None
    
    if audio_path:
        print(f"Audio prepared successfully: {audio_path}")
