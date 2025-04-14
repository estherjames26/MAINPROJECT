"""Audio and motion feature extraction functions for motion stitching."""
import numpy as np
import librosa
import torch

class AudioFeatureExtractor:
    def __init__(self, sr=22050, n_fft=2048, hop_length=512):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
    
    def extract_features(self, audio_path, beat_frames=None):
        y, sr = librosa.load(audio_path, sr=self.sr)
        if beat_frames is None:
            tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=self.n_fft, hop_length=self.hop_length)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length)
        
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=self.hop_length)
        
        features = {
            'mfcc': mfcc, 
            'mfcc_delta': mfcc_delta, 
            'mfcc_delta2': mfcc_delta2,
            'spectral_centroid': spectral_centroid,
            'spectral_contrast': spectral_contrast,
            'spectral_rolloff': spectral_rolloff,
            'chroma': chroma,
            'beat_frames': beat_frames,
            'beat_times': beat_times,
            'tempo': tempo
        }
        
        return features
    
    def get_beat_aligned_features(self, features, n_frames_per_beat=60):
        beat_frames = features['beat_frames']
        align_features = {}
        
        times = librosa.frames_to_time(np.arange(features['mfcc'].shape[1]), 
                                      sr=self.sr, hop_length=self.hop_length)
        
        beat_features = []
        for i in range(len(beat_frames) - 1):
            start_beat = beat_frames[i]
            end_beat = beat_frames[i+1]
            
            start_time = librosa.frames_to_time(start_beat, sr=self.sr, hop_length=self.hop_length)
            end_time = librosa.frames_to_time(end_beat, sr=self.sr, hop_length=self.hop_length)
            
            start_idx = np.argmin(np.abs(times - start_time))
            end_idx = np.argmin(np.abs(times - end_time))
            
            if end_idx <= start_idx:
                continue
            
            beat_feature = {}
            for feat_name, feat_matrix in features.items():
                if feat_name in ['beat_frames', 'beat_times', 'tempo']:
                    continue
                
                segment = feat_matrix[:, start_idx:end_idx]
                if segment.shape[1] > 0:
                    resampled = librosa.util.fix_length(segment, size=n_frames_per_beat, axis=1)
                    beat_feature[feat_name] = resampled
            
            if len(beat_feature) > 0:
                beat_features.append(beat_feature)
        
        return beat_features

class MotionFeatureExtractor:
    def __init__(self, window_size=120, step_size=30):
        self.window_size = window_size
        self.step_size = step_size
    
    def extract_windows(self, motion_sequence):
        seq_length = motion_sequence.shape[0]
        windows = []
        
        for start_idx in range(0, seq_length - self.window_size, self.step_size):
            end_idx = start_idx + self.window_size
            window = motion_sequence[start_idx:end_idx]
            windows.append(window)
        
        return windows
    
    def compute_velocity(self, motion_sequence):
        velocity = np.diff(motion_sequence, axis=0)
        velocity = np.vstack([velocity[0:1], velocity])
        return velocity
    
    def compute_acceleration(self, motion_sequence):
        velocity = self.compute_velocity(motion_sequence)
        acceleration = self.compute_velocity(velocity)
        return acceleration

def compute_motion_compatibility(motion1, motion2, overlap_frames=30):
    if motion1.shape[0] < overlap_frames or motion2.shape[0] < overlap_frames:
        return 0.0
    
    end1 = motion1[-overlap_frames:]
    start2 = motion2[:overlap_frames]
    
    velocity1 = np.diff(end1, axis=0)
    velocity2 = np.diff(start2, axis=0)
    
    pose_diff = np.mean(np.square(end1[-1] - start2[0]))
    velocity_diff = np.mean(np.square(velocity1[-1] - velocity2[0]))
    
    score = pose_diff + 0.5 * velocity_diff
    compatibility = np.exp(-score)
    
    return compatibility
