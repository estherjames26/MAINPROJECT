"""Database for storing and retrieving motion clips."""
import os
import joblib
import numpy as np
import random

class MotionDatabase:
    def __init__(self, database_path, config=None):
        """Initialize the motion database."""
        self.database_path = database_path
        self.config = config
        self.clips = {}
        self.metadata = {}
        self.loaded = False
        
        # Create the database directory if it doesn't exist
        os.makedirs(os.path.dirname(database_path), exist_ok=True)
    
    def load(self):
        """Load the database from disk."""
        if os.path.exists(self.database_path):
            try:
                data = joblib.load(self.database_path)
                self.clips = data.get('clips', {})
                self.metadata = data.get('metadata', {})
                print(f"Loaded {len(self.clips)} clips from database")
                self.loaded = True
                return True
            except Exception as e:
                print(f"Error loading database: {e}")
                return False
        else:
            print(f"Database file not found at {self.database_path}")
            return False
    
    def save(self):
        """Save the database to disk."""
        try:
            data = {
                'clips': self.clips,
                'metadata': self.metadata
            }
            joblib.dump(data, self.database_path, compress=3)
            print(f"Saved {len(self.clips)} clips to database")
            return True
        except Exception as e:
            print(f"Error saving database: {e}")
            return False
    
    def add_clip(self, clip_id, motion_data, metadata=None):
        """Add a motion clip to the database."""
        if clip_id in self.clips:
            print(f"Warning: Overwriting existing clip {clip_id}")
        
        self.clips[clip_id] = motion_data
        if metadata:
            self.metadata[clip_id] = metadata
        return True
    
    def get_clip(self, clip_id):
        """Retrieve a motion clip and its metadata from the database."""
        clip = self.clips.get(clip_id)
        meta = self.metadata.get(clip_id, {})
        return clip, meta
    
    def get_all_clips(self):
        """Get all clip IDs and basic info."""
        return self.clips.keys()
    
    def get_random_clip(self):
        """Get a random clip from the database."""
        if not self.clips:
            return None, None
        
        clip_id = random.choice(list(self.clips.keys()))
        return clip_id, self.clips[clip_id]
    
    def get_clips_info(self):
        """Get basic info about all clips."""
        info = {}
        for clip_id in self.clips:
            clip = self.clips[clip_id]
            
            # Get number of dancers and frames
            try:
                # Handle different motion data formats
                if isinstance(clip, dict):
                    # Dictionary format (AIST++ style)
                    if 'smpl_poses' in clip:
                        motion = clip['smpl_poses']
                    elif 'poses' in clip:
                        motion = clip['poses']
                    elif 'motion' in clip:
                        motion = clip['motion']
                    else:
                        # Unknown format, just record that it's a dict
                        info[clip_id] = {
                            'dancers': 'unknown',
                            'frames': 'unknown',
                            'format': 'dict',
                            'metadata': self.metadata.get(clip_id, {})
                        }
                        continue
                else:
                    # Direct numpy array
                    motion = clip
                
                # Determine shape from motion array
                if isinstance(motion, np.ndarray):
                    if len(motion.shape) == 3:  # [dancers, frames, features]
                        dancers = motion.shape[0]
                        frames = motion.shape[1]
                    else:  # [frames, features]
                        dancers = 1
                        frames = motion.shape[0]
                    
                    info[clip_id] = {
                        'dancers': dancers,
                        'frames': frames,
                        'metadata': self.metadata.get(clip_id, {})
                    }
                else:
                    # Not a numpy array
                    info[clip_id] = {
                        'dancers': 'unknown',
                        'frames': 'unknown',
                        'format': type(motion).__name__,
                        'metadata': self.metadata.get(clip_id, {})
                    }
            except Exception as e:
                # If any error occurs, record basic info
                info[clip_id] = {
                    'dancers': 'unknown',
                    'frames': 'unknown',
                    'error': str(e),
                    'metadata': self.metadata.get(clip_id, {})
                }
        
        return info
    
    def filter_clips(self, dancer_count=None):
        """Get clips matching criteria."""
        results = []
        
        for clip_id, clip in self.clips.items():
            # Check dancer count if specified
            if dancer_count is not None:
                try:
                    # Handle different motion data formats
                    if isinstance(clip, dict):
                        # Dictionary format (AIST++ style)
                        if 'smpl_poses' in clip:
                            motion = clip['smpl_poses']
                        elif 'poses' in clip:
                            motion = clip['poses']
                        elif 'motion' in clip:
                            motion = clip['motion']
                        else:
                            # Unknown format, skip this clip
                            continue
                    else:
                        # Direct numpy array
                        motion = clip
                    
                    # Check if dancer count matches
                    if isinstance(motion, np.ndarray):
                        if len(motion.shape) == 3 and motion.shape[0] != dancer_count:
                            continue
                        if len(motion.shape) == 2 and dancer_count != 1:
                            continue
                    else:
                        # Not a numpy array, skip this clip
                        continue
                except:
                    # If any error occurs, skip this clip
                    continue
            
            results.append(clip_id)
        
        return results
