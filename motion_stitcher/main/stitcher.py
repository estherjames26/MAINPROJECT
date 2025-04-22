# Motion stitching system for creating dance choreography
import os
import numpy as np
import pickle
import random 
import librosa
from typing import List, Dict, Tuple, Any, Optional
from motion_stitcher.main.features import compute_motion_compatibility

class MotionStitcher:
    #System for stitching together motion clips to create choreography

    def __init__(self, config, database=None):
        self.config = config
        self.database = database

    def stitch_choreography(self, audio_path, num_dancers, target_duration=None, style=None):
        #Create a choreography by stitching motion clips to match audio
        print(f"Creating choreography for {num_dancers} dancers with audio: {audio_path}")

        audio_features = self._extract_audio_features(audio_path)
        beat_times = audio_features.get("beat_times", [])
        duration = audio_features.get("duration", 60)

        MAX_DURATION_FRAMES = 450  # 15 seconds at 30 FPS
        if target_duration is None:
            target_duration = MAX_DURATION_FRAMES
            #print(f"Target duration not provided. Using max duration: {target_duration} frames")
        else:
            target_duration = min(target_duration, MAX_DURATION_FRAMES)
            print(f"Target duration capped to {target_duration} frames")

        beat_frames = [int(t * 30) for t in beat_times if t * 30 < target_duration]
        if not beat_frames:
            beat_frames = list(range(0, target_duration, 30))
            print("No beat timings detected, using uniform 30fps intervals.")

        clip_ids = self.database.filter_clips(dancer_count=num_dancers)
        if not clip_ids:
            print(f"No clips found with {num_dancers} dancers")
            return None

        if style:
            filtered_clips = []
            for clip_id in clip_ids:
                clip, meta = self.database.get_clip(clip_id)
                if meta.get('style') == style:
                    filtered_clips.append(clip_id)
            if filtered_clips:
                clip_ids = filtered_clips
                print(f"Filtered to {len(clip_ids)} clips with style: {style}")


        # Randomly select a starting clip as a seed motion
        current_clip_id = random.choice(clip_ids)
        current_clip_data, _ = self.database.get_clip(current_clip_id)
        current_clip = self._extract_motion_data(current_clip_data)

        choreography = current_clip.copy()
        if num_dancers > 1 and choreography.ndim != 3:
            print("Warning: Expected group choreography, but got solo motion format")

        current_frame = choreography.shape[0] if num_dancers == 1 else choreography.shape[1]
        beat_index = 1

        while beat_index < len(beat_frames) and current_frame < target_duration:
            target_frame = beat_frames[beat_index]
            beat_index += 1
            frames_needed = target_frame - current_frame + 30
            if frames_needed <= 0:
                continue

            next_clip_id, next_clip = self._find_compatible_clip(current_clip, clip_ids, num_dancers)
            if next_clip is None:
                print("No compatible clip found, ending choreography")
                break

            choreography, overlap = self._stitch_clips(choreography, next_clip, num_dancers)
            current_frame = choreography.shape[0] if num_dancers == 1 else choreography.shape[1]
            print(f"Stitched clip {next_clip_id} at frame {current_frame-overlap}, new length: {current_frame}")

        print(f"Choreography created with {current_frame} frames")

        metadata = {
            'audio_path': audio_path,
            'num_dancers': num_dancers,
            'style': style,
            'beat_times': beat_times,
            'duration': duration
        }

        smpl_poses = choreography
        smpl_trans = np.zeros((current_frame, 3)) if num_dancers == 1 else np.zeros((num_dancers, current_frame, 3))

        return {
            'motion': choreography,
            'metadata': metadata,
            'smpl_poses': smpl_poses,
            'smpl_trans': smpl_trans
        }

    def _extract_audio_features(self, audio_path):
        try:
            y, sr = librosa.load(audio_path, sr=22050)
            duration = librosa.get_duration(y=y, sr=sr)
            tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
            beat_times = librosa.frames_to_time(beat_frames, sr=sr)
            return {'duration': duration, 'beat_times': beat_times.tolist()}
        except Exception as e:
            print(f"Error extracting audio features: {e}")
            return {'duration': 60, 'beat_times': []}

    def _extract_motion_data(self, motion_data):
        try:
            if isinstance(motion_data, dict):
                if 'smpl_poses' in motion_data:
                    return motion_data['smpl_poses']
                elif 'motion' in motion_data:
                    return motion_data['motion']
            elif isinstance(motion_data, np.ndarray):
                return motion_data
            return motion_data
        except Exception as e:
            print(f"Error extracting motion data: {e}")
            return None

    def _find_compatible_clip(self, current_clip, clip_ids, num_dancers):
        best_clip_id = None
        best_clip = None
        best_score = -1
        overlap_frames = 30

        if num_dancers == 1:
            current_end = current_clip[-overlap_frames:] if current_clip.shape[0] >= overlap_frames else current_clip
        else:
            current_end = current_clip[:, -overlap_frames:] if current_clip.shape[1] >= overlap_frames else current_clip

        candidates = random.sample(clip_ids, min(10, len(clip_ids)))

        for clip_id in candidates:
            clip_data, _ = self.database.get_clip(clip_id)
            clip = self._extract_motion_data(clip_data)
            if clip is None:
                continue

            if num_dancers == 1:
                if clip.shape[0] < overlap_frames:
                    continue
                clip_start = clip[:overlap_frames]
                score = compute_motion_compatibility(current_end, clip_start, overlap_frames)
            else:
                if clip.shape[1] < overlap_frames:
                    continue
                clip_start = clip[:, :overlap_frames]
                scores = []
                for d in range(num_dancers):
                    dancer_score = compute_motion_compatibility(current_end[d, :], clip_start[d, :], overlap_frames)
                    scores.append(dancer_score)
                score = sum(scores) / len(scores)

            if score > best_score:
                best_score = score
                best_clip_id = clip_id
                best_clip = clip

        return best_clip_id, best_clip

    def _stitch_clips(self, clip1, clip2, num_dancers):
        overlap_frames = min(30, clip1.shape[1] if num_dancers > 1 else clip1.shape[0], 
                            clip2.shape[1] if num_dancers > 1 else clip2.shape[0])
        blend_frames = min(20, overlap_frames)

        if num_dancers == 1:
            result_length = clip1.shape[0] + clip2.shape[0] - overlap_frames
            result = np.zeros((result_length, clip1.shape[1]), dtype=clip1.dtype)
            result[:clip1.shape[0]] = clip1
            for i in range(blend_frames):
                blend_factor = i / blend_frames
                overlap_idx = clip1.shape[0] - blend_frames + i
                result_idx = overlap_idx
                result[result_idx] = (1 - blend_factor) * clip1[overlap_idx] + blend_factor * clip2[i]
            result_start_idx = clip1.shape[0]
            clip2_start_idx = blend_frames
            frames_to_copy = min(clip2.shape[0] - clip2_start_idx, result.shape[0] - result_start_idx)
            result[result_start_idx:result_start_idx+frames_to_copy] = clip2[clip2_start_idx:clip2_start_idx+frames_to_copy]
        else:
            result_length = clip1.shape[1] + clip2.shape[1] - overlap_frames
            result = np.zeros((num_dancers, result_length, clip1.shape[2]), dtype=clip1.dtype)
            result[:, :clip1.shape[1]] = clip1
            for d in range(num_dancers):
                for i in range(blend_frames):
                    blend_factor = i / blend_frames
                    overlap_idx = clip1.shape[1] - blend_frames + i
                    result_idx = overlap_idx
                    result[d, result_idx] = (1 - blend_factor) * clip1[d, overlap_idx] + blend_factor * clip2[d, i]
            result_start_idx = clip1.shape[1]
            clip2_start_idx = blend_frames
            frames_to_copy = min(clip2.shape[1] - clip2_start_idx, result.shape[1] - result_start_idx)
            if frames_to_copy > 0:
                result[:, result_start_idx:result_start_idx+frames_to_copy] = clip2[:, clip2_start_idx:clip2_start_idx+frames_to_copy]

        return result, overlap_frames

    def generate_choreography(self, audio_path, output_path, num_dancers=1, target_duration=None, style=None):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        choreography = self.stitch_choreography(audio_path, num_dancers, target_duration, style)
        if choreography is None:
            print("Failed to generate choreography")
            return False
        try:
            with open(output_path, 'wb') as f:
                pickle.dump(choreography, f)
            print(f"Saved choreography to {output_path}")
            return True
        except Exception as e:
            print(f"Error saving choreography: {e}")
            return False
