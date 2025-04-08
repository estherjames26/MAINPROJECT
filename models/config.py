"""Model configuration parameters."""

class Config:
    def __init__(self):
        # Data parameters
        self.seq_len = 120  # Reduced from 240 for faster training
        self.motion_dim = 75  # Per dancer (SMPL pose + translation)
        self.num_dancers = 3  # Fixed for now
        self.max_dancers = 3  # Maximum number of dancers supported
        self.audio_dim = 28  # MFCC + chroma + onset + beat
        
        # Model parameters - enhanced for better performance
        self.hidden_dim = 64   # Increased from 32 for better representation
        self.num_layers = 2    # Increased from 1 for better temporal modeling
        self.num_heads = 4     # Increased from 2 for better attention
        self.ff_size = 256     # Increased from 128 for better capacity
        self.dropout = 0.1
        
        # Embedding dimensions - enhanced for better representation
        self.music_embedding_dim = 64   # Increased for better music understanding
        self.motion_embedding_dim = 64  # Increased for better motion representation
        self.latent_dim = 64            # Increased for better latent space
        self.condition_dim = 16         # Increased for better conditioning
        
        # Training parameters - optimized for better convergence
        self.batch_size = 32    # Keep batch size large for diversity
        self.learning_rate = 0.001  # Reduced further for better stability
        self.weight_decay = 1e-5   # Increased for better regularization
        self.max_epochs = 1000   # Keep high max epochs with early stopping
        self.warmup_epochs = 5   # Increased for better initialization
        self.gradient_clip = 1.0
        self.gradient_accumulation_steps = 1
        self.lr_decay = True    # Enable learning rate decay
        self.lr_decay_factor = 0.95  # Decay factor for learning rate
        self.lr_decay_epochs = 10    # Apply decay every 10 epochs
        
        # Loss weights - comprehensive loss function
        self.w_recon = 1.0       # Base reconstruction weight
        self.w_kld = 0.001       # Small KL weight for VAE regularization
        self.w_velocity = 0.1    # Velocity matching for smoother motion
        self.w_acceleration = 0.05  # Acceleration matching for natural motion
        self.w_sync = 0.2        # Synchronization between dancers
        self.w_smoothness = 0.1  # Additional smoothness term (jerk minimization)
        self.w_beat = 0.3        # Beat alignment weight
        self.w_diversity = 0.05  # Diversity regularization
        
        # Evaluation parameters
        self.eval_beat_idx = -1  # Index of beat feature in audio
        self.eval_frequency = 1  # Evaluate every N epochs
        
        # Device settings
        self.use_cuda = True
        self.num_workers = 0  # Reduced to 0 to avoid data loading overhead
        self.pin_memory = True
        
        # Model specific
        self.use_style_condition = False
        self.use_genre_condition = False
        self.num_styles = 10
        self.num_genres = 10
        
        # Memory optimization
        self.use_gradient_checkpointing = False  # Disabled for faster training
        
        # Save directory
        self.save_dir = "results"  # Directory to save models and results
