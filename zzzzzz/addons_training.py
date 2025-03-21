import tensorflow as tf

def ensure_orthogonal_rotmats(rotmats):
    """Ensure rotation matrices are orthogonal during training"""
    if isinstance(rotmats, tf.Tensor):
        # For TensorFlow tensors
        batch_size = tf.shape(rotmats)[0]
        seq_len = tf.shape(rotmats)[1]
        num_joints = 24
        rotmats_reshaped = tf.reshape(rotmats[:, :, 3:], [-1, num_joints, 3, 3])
        
        # Process each joint's rotation matrix
        stabilized_rotmats = []
        for j in range(num_joints):
            # Extract rotation matrices for this joint
            joint_rotmats = rotmats_reshaped[:, j]
            
            # Use SVD to find the closest orthogonal matrix
            s, u, v = tf.linalg.svd(joint_rotmats)
            joint_rotmats_ortho = tf.matmul(u, v, transpose_b=True)
            
            stabilized_rotmats.append(joint_rotmats_ortho)
        
        # Stack joints back together
        stabilized_rotmats = tf.stack(stabilized_rotmats, axis=1)
        
        # Reshape back and combine with translation
        stabilized_rotmats_flat = tf.reshape(stabilized_rotmats, [batch_size, seq_len, -1])
        trans = rotmats[:, :, :3]
        
        return tf.concat([trans, stabilized_rotmats_flat], axis=2)
    else:
        # For numpy arrays - similar process but with numpy functions
        return rotmats
    


def orthogonal_loss(motion_output):
    """Add a penalty term to encourage orthogonal rotation matrices"""
    batch_size = tf.shape(motion_output)[0]
    seq_len = tf.shape(motion_output)[1]
    
    # Extract rotation matrices (reshape from flattened form)
    rot_matrices = tf.reshape(motion_output[:, :, 3:], [batch_size, seq_len, 24, 3, 3])
    
    # For each rotation matrix R, compute ||R^T·R - I||
    identity = tf.eye(3, batch_shape=[batch_size, seq_len, 24])
    rot_transpose = tf.transpose(rot_matrices, perm=[0, 1, 2, 4, 3])
    orthogonality = tf.matmul(rot_transpose, rot_matrices)
    
    # Compute the loss as the deviation from identity
    orth_loss = tf.reduce_mean(tf.square(orthogonality - identity))
    
    return orth_loss

def custom_loss(y_true, y_pred):
    # Regular MSE loss
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    
    # Add orthogonality constraint with a small weight
    orth_loss = orthogonal_loss(y_pred) * 0.01
    
    return mse_loss + orth_loss

