import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.signal import savgol_filter


def stabilize_rotation_matrix(rotation_matrix):
    """Stabilize a rotation matrix to ensure it's orthogonal"""
    # Use SVD to find the closest orthogonal matrix
    u, _, vh = np.linalg.svd(rotation_matrix, full_matrices=False)
    return u @ vh

def convert_219_to_75(frame_219: np.ndarray) -> np.ndarray:
    """
    Converts a single FACT output frame (219D) into a SMPL-compatible frame (75D).
    Includes stabilization for rotation matrices.
    """
    if frame_219.shape[0] != 219:
        raise ValueError(f"Expected frame dimension 219, got {frame_219.shape[0]}")
    
    trans = frame_219[:3]
    rot_flat = frame_219[3:]
    rot_mats = rot_flat.reshape((24, 3, 3))
    
    # Stabilize each rotation matrix before conversion
    for i in range(24):
        rot_mats[i] = stabilize_rotation_matrix(rot_mats[i])
    
    axis_angles = R.from_matrix(rot_mats).as_rotvec().reshape(-1)
    return np.concatenate([trans, axis_angles], axis=0)

def apply_joint_limits(smpl_poses):
    """Apply anatomical joint limits to SMPL poses to prevent unnatural poses"""
    # Define limits for each joint in axis-angle space
    joint_limits = {
        # Neck
        1: {'x': (-0.7, 0.7), 'y': (-0.7, 0.7), 'z': (-0.7, 0.7)},
        # Spine
        3: {'x': (-0.4, 0.4), 'y': (-0.4, 0.4), 'z': (-0.4, 0.4)},
        # Shoulders
        13: {'x': (-np.pi/2, np.pi/2), 'y': (-np.pi/2, np.pi/2), 'z': (-np.pi/2, np.pi/2)},
        16: {'x': (-np.pi/2, np.pi/2), 'y': (-np.pi/2, np.pi/2), 'z': (-np.pi/2, np.pi/2)},
        # Elbows (limited bend)
        14: {'x': (-0.1, 0.1), 'y': (-np.pi, 0.1), 'z': (-0.1, 0.1)},
        17: {'x': (-0.1, 0.1), 'y': (0, np.pi), 'z': (-0.1, 0.1)},
        # Knees (only bend in one direction)
        4: {'x': (-0.1, 0.1), 'y': (0, np.pi), 'z': (-0.1, 0.1)},
        5: {'x': (-0.1, 0.1), 'y': (0, np.pi), 'z': (-0.1, 0.1)}
    }
    
    limited_poses = smpl_poses.copy()
    for joint_idx, limits in joint_limits.items():
        # Apply x limit
        limited_poses[:, joint_idx*3] = np.clip(
            limited_poses[:, joint_idx*3], limits['x'][0], limits['x'][1])
        # Apply y limit
        limited_poses[:, joint_idx*3+1] = np.clip(
            limited_poses[:, joint_idx*3+1], limits['y'][0], limits['y'][1])
        # Apply z limit
        limited_poses[:, joint_idx*3+2] = np.clip(
            limited_poses[:, joint_idx*3+2], limits['z'][0], limits['z'][1])
    
    return limited_poses

def smooth_motion(motion_data, window_size=7):
    """Apply temporal smoothing to reduce jitter"""
    from scipy.signal import savgol_filter
    smoothed = np.zeros_like(motion_data)
    
    # Make sure window size is odd
    if window_size % 2 == 0:
        window_size += 1
    
    # Make sure we have enough frames for the window
    if motion_data.shape[0] < window_size:
        print(f"Warning: Not enough frames for smoothing window of size {window_size}. Using smaller window.")
        window_size = min(5, motion_data.shape[0])
        if window_size % 2 == 0 and window_size > 1:
            window_size -= 1
    
    if window_size < 3:
        print("Not enough frames for smoothing. Returning original motion.")
        return motion_data
    
    # Apply Savitzky-Golay filter to each dimension
    for i in range(motion_data.shape[1]):
        smoothed[:, i] = savgol_filter(motion_data[:, i], window_size, 2)
    
    return smoothed

def smooth(motion_data, window_size=7):
    """Apply temporal smoothing to reduce jitter"""
    if window_size % 2 == 0:
        window_size += 1  # Must be odd
    
    if motion_data.shape[0] < window_size:
        window_size = min(5, motion_data.shape[0])
        if window_size % 2 == 0 and window_size > 1:
            window_size -= 1
    
    if window_size < 3:
        return motion_data
    
    smoothed = np.zeros_like(motion_data)
    for i in range(motion_data.shape[1]):
        smoothed[:, i] = savgol_filter(motion_data[:, i], window_size, 2)
    
    return smoothed

def verify_bone_lengths(smpl_poses, threshold=0.5):
    """Check if bone lengths remain consistent across frames"""
    # This is a simplified check - ideally would use SMPL forward kinematics
    # but we can approximate by checking for rapid changes in joint velocities
    velocities = np.diff(smpl_poses, axis=0)
    max_velocities = np.max(np.abs(velocities), axis=0)
    
    problematic_joints = np.where(max_velocities > threshold)[0]
    if len(problematic_joints) > 0:
        print(f"Warning: Potential bone length inconsistencies at joints: {problematic_joints // 3}")
    
    return len(problematic_joints) == 0