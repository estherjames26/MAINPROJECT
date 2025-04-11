"""Motion stitching system for creating dance choreography."""
import os
import numpy as np
import pickle
import random
from typing import List, Dict, Tuple, Any, Optional

class MotionStitcher:
    """System for stitching together motion clips to create choreography."""
    
    def __init__(self, config, database=None):
        self.config = config
        self.database = database
    
    def stitch_choreography(self, audio_path, num_dancers=1, target_duration=None, style=None):
        """Create a choreography by stitching motion clips to match audio."""
        print(f"Creating choreography for {num_dancers} dancers with audio: {audio_path}")
        
        # Basic audio duration estimate (will be improved later)
        audio_features = self._extract_audio_features(audio_path)
        
        if target_duration is None:
            target_duration = int(audio_features.get('duration', 60) * 30)  # 30 fps
            print(f"Target duration set to {target_duration} frames")
        
        # Get clips that match the dancer count
        clip_ids = self.database.filter_clips(dancer_count=num_dancers)
        if not clip_ids:
            print(f"No clips found with {num_dancers} dancers")
            return None
        
        # Filter by style if specified
        if style:
            filtered_clips = []
            for clip_id in clip_ids:
                clip, meta = self.database.get_clip(clip_id)
                if meta.get('style') == style:
                    filtered_clips.append(clip_id)
            
            if filtered_clips:
                clip_ids = filtered_clips
                print(f"Filtered to {len(clip_ids)} clips with style: {style}")
        
        # Start with a random clip
        current_clip_id = random.choice(clip_ids)
        current_clip_data, _ = self.database.get_clip(current_clip_id)
        
        # Extract motion data from current clip (may be dictionary or numpy array)
        current_clip = self._extract_motion_data(current_clip_data)
        
        # Initialize the choreography with the first clip
        if num_dancers == 1:
            choreography = current_clip.copy()
        else:
            choreography = current_clip.copy()
        
        # Stitch until target duration reached
        current_frame = choreography.shape[0] if num_dancers == 1 else choreography.shape[1]
        
        while current_frame < target_duration:
            # Find compatible next clip
            next_clip_id, next_clip = self._find_compatible_clip(current_clip, clip_ids, num_dancers)
            
            if next_clip is None:
                print("No compatible clip found, ending choreography")
                break
            
            # Stitch the next clip
            choreography, overlap = self._stitch_clips(choreography, next_clip, num_dancers)
            
            current_frame = choreography.shape[0] if num_dancers == 1 else choreography.shape[1]
            print(f"Stitched clip {next_clip_id} at frame {current_frame-overlap}, new length: {current_frame}")
        
        print(f"Choreography created with {current_frame} frames")
        
        # Add metadata
        metadata = {
            'audio_path': audio_path,
            'num_dancers': num_dancers,
            'style': style
        }
        
        return {
            'motion': choreography,
            'metadata': metadata
        }
    
    def _extract_audio_features(self, audio_path):
        """Extract basic duration from the audio file."""
        try:
            # Simple placeholder for audio features
            # In future versions, this will extract beats and other features
            return {'duration': 60}
        except Exception as e:
            print(f"Error extracting audio features: {e}")
            return {'duration': 60}
    
    def _extract_motion_data(self, motion_data):
        """Extract motion array from various formats."""
        try:
            if isinstance(motion_data, dict):
                # Dictionary format (AIST++ style)
                if 'smpl_poses' in motion_data:
                    return motion_data['smpl_poses']
                elif 'poses' in motion_data:
                    return motion_data['poses']
                elif 'motion' in motion_data:
                    return motion_data['motion']
                else:
                    raise ValueError("Unknown motion data format")
            elif isinstance(motion_data, np.ndarray):
                # Already a numpy array
                return motion_data
            else:
                raise ValueError(f"Unsupported motion data type: {type(motion_data)}")
        except Exception as e:
            print(f"Error extracting motion data: {e}")
            return None
    
    def _find_compatible_clip(self, current_clip, clip_ids, num_dancers):
        """Find a clip that is compatible with the current clip."""
        # Select random candidates for efficiency
        candidates = random.sample(clip_ids, min(5, len(clip_ids)))
        
        best_clip_id = None
        best_clip = None
        
        for clip_id in candidates:
            candidate_data, _ = self.database.get_clip(clip_id)
            if candidate_data is None:
                continue
            
            # Extract motion data
            candidate = self._extract_motion_data(candidate_data)
            if candidate is None:
                continue
            
            # Calculate compatibility score
            if num_dancers == 1:
                # Solo case: [frames, joints]
                score = np.mean(np.abs(current_clip[-30:] - candidate[:30]))
            else:
                # Group case: [dancers, frames, joints]
                scores = []
                for d in range(num_dancers):
                    score = np.mean(np.abs(current_clip[d, -30:] - candidate[d, :30]))
                    scores.append(score)
                score = np.mean(scores)  # Average across dancers
            
            if best_clip is None or score < np.mean(np.abs(best_clip)):
                best_clip_id = clip_id
                best_clip = candidate
        
        return best_clip_id, best_clip
    
    def _stitch_clips(self, clip1, clip2, num_dancers):
        """Stitch two clips together with blending in the overlap region."""
        overlap_frames = min(30, clip1.shape[1] if num_dancers > 1 else clip1.shape[0], 
                            clip2.shape[1] if num_dancers > 1 else clip2.shape[0])
        blend_frames = min(20, overlap_frames)
        
        if num_dancers == 1:
            # Handle single dancer case: [frames, joints]
            result_length = clip1.shape[0] + clip2.shape[0] - overlap_frames
            result = np.zeros((result_length, clip1.shape[1]), dtype=clip1.dtype)
            
            # Copy first clip completely
            result[:clip1.shape[0]] = clip1
            
            # Blend in the overlap region
            for i in range(blend_frames):
                blend_factor = i / blend_frames
                overlap_idx = clip1.shape[0] - blend_frames + i
                result_idx = overlap_idx
                
                result[result_idx] = (1 - blend_factor) * clip1[overlap_idx] + blend_factor * clip2[i]
            
            # Copy second clip after the blend region
            result_start_idx = clip1.shape[0] - blend_frames + blend_frames
            clip2_start_idx = blend_frames
            
            # Calculate how many frames to copy from clip2
            frames_to_copy = min(clip2.shape[0] - clip2_start_idx, result.shape[0] - result_start_idx)
            
            # Copy safely considering the available space
            result[result_start_idx:result_start_idx+frames_to_copy] = clip2[clip2_start_idx:clip2_start_idx+frames_to_copy]
            
        else:
            # Handle multi-dancer case: [dancers, frames, joints]
            result_length = clip1.shape[1] + clip2.shape[1] - overlap_frames
            result = np.zeros((num_dancers, result_length, clip1.shape[2]), dtype=clip1.dtype)
            
            # Copy first clip completely
            result[:, :clip1.shape[1]] = clip1
            
            # Blend in the overlap region for each dancer
            for d in range(num_dancers):
                for i in range(blend_frames):
                    blend_factor = i / blend_frames
                    overlap_idx = clip1.shape[1] - blend_frames + i
                    result_idx = overlap_idx
                    
                    result[d, result_idx] = (1 - blend_factor) * clip1[d, overlap_idx] + blend_factor * clip2[d, i]
            
            # Copy second clip after the blend region
            result_start_idx = clip1.shape[1] - blend_frames + blend_frames
            clip2_start_idx = blend_frames
            
            # Calculate how many frames to copy from clip2
            frames_to_copy = min(clip2.shape[1] - clip2_start_idx, result.shape[1] - result_start_idx)
            
            # Copy safely considering the available space
            if frames_to_copy > 0:
                result[:, result_start_idx:result_start_idx+frames_to_copy] = clip2[:, clip2_start_idx:clip2_start_idx+frames_to_copy]
        
        return result, overlap_frames
    
    def generate_choreography(self, audio_path, output_path, num_dancers=1, target_duration=None, style=None):
        """Generate a choreography and save it to a file."""
        # Check if output directory exists, create if not
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Generate the choreography
        choreography = self.stitch_choreography(audio_path, num_dancers, target_duration, style)
        
        if choreography is None:
            print("Failed to generate choreography")
            return False
        
        # Save the choreography
        try:
            with open(output_path, 'wb') as f:
                pickle.dump(choreography, f)
            print(f"Saved choreography to {output_path}")
            return True
        except Exception as e:
            print(f"Error saving choreography: {e}")
            return False
