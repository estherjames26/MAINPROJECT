"""Training script for dance generation model."""

import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR, StepLR
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from collections import defaultdict
import time
from datetime import datetime, timedelta

from models.config import Config
from models.layers import GroupMotionVAE
from models.losses import DanceGenerationLoss
from models.dataset import create_dataloader
from models.metrics import DanceEvaluator
from models.visualize import (plot_motion_features, plot_beat_alignment,
                      create_motion_animation, save_evaluation_plots)
from pathlib import Path

def train_epoch(model, train_loader, criterion, optimizer, device, epoch, config):
    model.train()
    total_loss = 0
    loss_components = defaultdict(float)
    
    # Start timing the epoch
    start_time = time.time()
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(progress_bar):
        # Move batch to device
        motion = batch['motion'].to(device, non_blocking=True)
        audio = batch['audio'].to(device, non_blocking=True)
        num_dancers = batch['num_dancers'][0]
        
        # Forward pass
        pred_motion, mu, logvar = model(audio, motion, num_dancers)
        
        # Calculate loss - pass audio for beat alignment
        loss, components = criterion(pred_motion, motion, mu, logvar, num_dancers, audio)
        
        # Scale loss for gradient accumulation (if used)
        if config.gradient_accumulation_steps > 1:
            loss = loss / config.gradient_accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Only update weights after accumulating gradients for specified steps
        if config.gradient_accumulation_steps == 1 or (batch_idx + 1) % config.gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            
            # Update weights
            optimizer.step()
            optimizer.zero_grad()
        
        # Update metrics
        batch_loss = loss.item()
        if config.gradient_accumulation_steps > 1:
            batch_loss *= config.gradient_accumulation_steps  # Scale back for reporting
        total_loss += batch_loss
        for k, v in components.items():
            loss_components[k] += v
        
        # Calculate elapsed and remaining time
        elapsed = time.time() - start_time
        progress = (batch_idx + 1) / len(train_loader)
        if progress > 0:
            remaining = elapsed / progress - elapsed
        else:
            remaining = 0
        
        # Update progress bar with time information and key loss components
        progress_info = {
            'loss': f"{batch_loss:.4f}",
            'recon': f"{components['recon']:.4f}"
        }
        
        # Add other important components if they exist
        for key in ['beat', 'sync', 'smooth']:
            if key in components:
                progress_info[key] = f"{components[key]:.4f}"
                
        progress_info.update({
            'elapsed': f"{elapsed:.1f}s",
            'remain': f"{remaining:.1f}s"
        })
        
        progress_bar.set_postfix(progress_info)
    
    # Calculate averages
    num_batches = len(train_loader)
    avg_loss = total_loss / num_batches
    avg_components = {k: v / num_batches for k, v in loss_components.items()}
    
    # Calculate epoch time
    epoch_time = time.time() - start_time
    
    return avg_loss, avg_components, epoch_time

@torch.no_grad()
def evaluate(model, val_loader, criterion, evaluator, device, save_dir=None):
    model.eval()
    total_loss = 0
    loss_components = defaultdict(float)
    evaluator.reset()
    
    # Start timing the evaluation
    start_time = time.time()
    
    # Create visualization directory if it doesn't exist
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    for batch in val_loader:
        # Move data to device
        motion = batch['motion'].to(device, non_blocking=True)
        audio = batch['audio'].to(device, non_blocking=True)
        num_dancers = batch['num_dancers'][0]
        
        # Generate motion
        pred_motion, mu, logvar = model(audio, motion, num_dancers)
        
        # Calculate loss - pass audio for beat alignment
        loss, components = criterion(pred_motion, motion, mu, logvar, num_dancers, audio)
        
        # Update metrics
        total_loss += loss.item()
        for k, v in components.items():
            loss_components[k] += v
        
        # Update evaluator with predicted and target motion
        evaluator.update(pred_motion, motion, audio, num_dancers)
    
    # Calculate averages
    num_batches = len(val_loader)
    avg_loss = total_loss / num_batches
    avg_components = {k: v / num_batches for k, v in loss_components.items()}
    
    # Compute evaluation metrics
    metrics = evaluator.compute()
    
    # Calculate evaluation time
    eval_time = time.time() - start_time
    
    return avg_loss, avg_components, metrics, eval_time

def create_test_data():
    """Create small test dataset for initial testing."""
    import os
    import numpy as np
    import pandas as pd
    from pathlib import Path
    
    try:
        # Create directories
        base_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        test_dir = base_dir / 'test_data'
        data_dir = test_dir / 'preprocessed' / 'aioz'
        labels_dir = test_dir / 'data' / 'aioz'
        
        # Create fresh directories with explicit permissions
        data_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate sample data
        num_samples = 100
        seq_len = 240
        motion_dim = 75
        audio_dim = 28
        num_dancers = 3
        
        for split in ['train', 'val', 'test']:
            # Determine number of samples for this split
            if split == 'train':
                n = int(num_samples * 0.7)
            elif split == 'val':
                n = int(num_samples * 0.15)
            else:
                n = int(num_samples * 0.15)
            
            # Generate sample IDs
            sample_ids = [f"test_{split}_{i:03d}" for i in range(n)]
            
            # Create label file
            df = pd.DataFrame({
                'id': sample_ids,
                'music_genre': ['pop'] * n,
                'dance_style': ['modern'] * n
            })
            
            label_file = labels_dir / f"{split}_labels.csv"
            print(f"Writing labels to {label_file}...")
            df.to_csv(str(label_file), index=False)
            
            # Generate motion and audio data
            for sample_id in sample_ids:
                # Random motion data
                motion = np.random.randn(seq_len, num_dancers * motion_dim).astype(np.float32)
                motion_path = data_dir / f"{sample_id}_motion.npy"
                print(f"Writing motion to {motion_path}...")
                np.save(str(motion_path), motion)
                
                # Random audio features
                audio = np.random.randn(seq_len, audio_dim).astype(np.float32)
                audio_path = data_dir / f"{sample_id}_audio.npy"
                print(f"Writing audio to {audio_path}...")
                np.save(str(audio_path), audio)
        
        print("\nCreated test dataset with:")
        print(f"- Training samples: {int(num_samples * 0.7)}")
        print(f"- Validation samples: {int(num_samples * 0.15)}")
        print(f"- Test samples: {int(num_samples * 0.15)}")
        print(f"Data directory: {data_dir}")
        print(f"Labels directory: {labels_dir}")
        
        return True, data_dir, labels_dir
        
    except Exception as e:
        print(f"Error creating test data: {str(e)}")
        return False, None, None

def setup_device(config):
    """Setup the device for training."""
    if config.use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"- CUDA Version: {torch.version.cuda}")
        print(f"- Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device('cpu')
        print("Using CPU - No GPU available")
    return device

def main():
    # Load configuration
    config = Config()
    
    # Setup device
    device = setup_device(config)
    
    # Create save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(config.save_dir) / f"dance_model_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create test data if needed
    create_test_data()
    
    # Create data loaders
    train_loader, val_loader = create_dataloader(config)
    
    # Create model
    model = GroupMotionVAE(config).to(device)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create loss function
    criterion = DanceGenerationLoss(config)
    
    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Create learning rate scheduler
    if config.lr_decay:
        scheduler = StepLR(
            optimizer,
            step_size=config.lr_decay_epochs,
            gamma=config.lr_decay_factor
        )
    else:
        scheduler = None
    
    # Create evaluator
    evaluator = DanceEvaluator(config)
    
    # Create tensorboard writer
    writer = SummaryWriter(log_dir=str(save_dir / 'logs'))
    
    # Training loop
    best_loss = float('inf')
    patience = 20  # Early stopping patience
    patience_counter = 0
    
    train_losses = []
    val_losses = []
    epoch_times = []
    
    # Start timing total training
    total_start_time = time.time()
    
    print(f"Starting training with config:")
    for key, value in vars(config).items():
        print(f"  {key}: {value}")
    
    for epoch in range(config.max_epochs):
        print(f"\nEpoch {epoch+1}/{config.max_epochs}")
        
        # Train for one epoch
        train_loss, train_components, train_time = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, config
        )
        train_losses.append(train_loss)
        
        # Evaluate model
        val_loss, val_components, val_metrics, eval_time = evaluate(
            model, val_loader, criterion, evaluator, device, 
            save_dir=str(save_dir / 'visualizations') if epoch % 10 == 0 else None
        )
        val_losses.append(val_loss)
        
        # Update learning rate
        if scheduler:
            current_lr = optimizer.param_groups[0]['lr']
            scheduler.step()
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr != current_lr:
                print(f"Learning rate updated: {current_lr:.6f} -> {new_lr:.6f}")
        else:
            current_lr = optimizer.param_groups[0]['lr']
        
        # Record epoch time
        epoch_time = train_time + eval_time
        epoch_times.append(epoch_time)
        
        # Calculate average epoch time and estimate remaining time
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        remaining_epochs = config.max_epochs - (epoch + 1)
        estimated_remaining_time = avg_epoch_time * remaining_epochs
        
        # Format time estimates
        elapsed_time = time.time() - total_start_time
        estimated_total_time = elapsed_time + estimated_remaining_time
        
        # Print training progress with time estimates
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Print key loss components
        print("Loss Components:")
        for k in ['recon', 'vel', 'acc', 'kld', 'sync', 'smooth', 'beat', 'div']:
            if k in train_components:
                print(f"  {k}: train={train_components[k]:.4f}, val={val_components.get(k, 0):.4f}")
        
        # Print key metrics
        print("Evaluation Metrics:")
        for k in ['beat_alignment', 'smoothness', 'synchronization', 'diversity']:
            if k in val_metrics:
                print(f"  {k}: {val_metrics[k]:.4f}")
        
        print(f"Epoch time: {epoch_time:.2f}s (Train: {train_time:.2f}s, Eval: {eval_time:.2f}s)")
        print(f"Time elapsed: {timedelta(seconds=int(elapsed_time))}, Remaining: {timedelta(seconds=int(estimated_remaining_time))}")
        print(f"Estimated completion: {datetime.now() + timedelta(seconds=int(estimated_remaining_time))}")
        
        # Calculate improvement percentage
        if epoch > 0:
            train_improvement = (train_losses[-2] - train_loss) / train_losses[-2] * 100
            val_improvement = (val_losses[-2] - val_loss) / val_losses[-2] * 100
            print(f"Train improvement: {train_improvement:.2f}%, Val improvement: {val_improvement:.2f}%")
        
        # Log metrics
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('LearningRate', current_lr, epoch)
        writer.add_scalar('Time/epoch', epoch_time, epoch)
        writer.add_scalar('Time/train', train_time, epoch)
        writer.add_scalar('Time/eval', eval_time, epoch)
        
        # Log loss components
        for k, v in train_components.items():
            writer.add_scalar(f'Components/{k}_train', v, epoch)
        for k, v in val_components.items():
            writer.add_scalar(f'Components/{k}_val', v, epoch)
            
        # Log evaluation metrics
        for k, v in val_metrics.items():
            writer.add_scalar(f'Metrics/{k}', v, epoch)
        
        # Save best model
        if val_loss < best_loss:
            improvement = (best_loss - val_loss) / best_loss * 100
            best_loss = val_loss
            patience_counter = 0
            
            save_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict() if not isinstance(model, nn.DataParallel) 
                                  else model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'best_loss': best_loss,
                'config': config,
                'metrics': val_metrics
            }
            torch.save(save_dict, str(save_dir / 'best_model.pth'))
            print(f"Saved new best model with loss: {best_loss:.4f} (improved by {improvement:.2f}%)")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epochs (best: {best_loss:.4f})")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            save_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict() if not isinstance(model, nn.DataParallel)
                                  else model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'loss': val_loss,
                'config': config,
                'metrics': val_metrics
            }
            torch.save(save_dict, str(save_dir / f'checkpoint_epoch_{epoch}.pth'))
            
        # Explicitly clear CUDA cache after each epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Calculate total training time
    total_training_time = time.time() - total_start_time
    
    writer.close()
    print("\nTraining complete!")
    print(f"Best validation loss: {best_loss:.4f}")
    print(f"Total training time: {timedelta(seconds=int(total_training_time))}")
    print(f"Average epoch time: {sum(epoch_times) / len(epoch_times):.2f}s")
    print(f"Model saved to: {save_dir}")

if __name__ == '__main__':
    main()
