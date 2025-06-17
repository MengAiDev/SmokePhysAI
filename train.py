import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

import yaml
import os
import argparse

from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from typing import Dict

from src.utils.data_loader import DataLoader, create_data_loaders
from src.models.physics_regularizer import PhysicsRegularizer


def load_config(config_path: str) -> dict:
    """Load configuration file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def setup_experiment(config: dict) -> tuple:
    """Set up experiment environment"""
    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join('experiments', f'smokephys_{timestamp}')
    os.makedirs(exp_dir, exist_ok=True)
    
    # Set up tensorboard
    writer = SummaryWriter(os.path.join(exp_dir, 'logs'))
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    return exp_dir, writer, device

def train_epoch(model: nn.Module,
               train_loader: DataLoader,
               optimizer: optim.Optimizer,
               physics_regularizer: PhysicsRegularizer,
               device: str,
               epoch: int,
               writer: SummaryWriter) -> Dict[str, float]:
    """Train one epoch"""
    model.train()
    
    total_loss = 0.0
    total_recon_loss = 0.0
    total_physics_loss = 0.0
    total_chaos_loss = 0.0
    
    pbar = tqdm(train_loader, desc=f'Training Epoch {epoch+1}', leave=True)
    for batch_idx, batch in enumerate(pbar):
        # Move data to device
        inputs = batch['input'].to(device)
        targets = batch['target'].to(device)
        chaos_targets = batch['chaos_features'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        
        # Reconstruction loss
        recon_loss = F.mse_loss(outputs['reconstructed'], targets)
        
        # Chaos feature loss
        chaos_loss = F.mse_loss(outputs['physics_features'], chaos_targets)
        
        # Physics regularization
        physics_losses = physics_regularizer({
            'density': outputs['reconstructed'],
            'density_sequence': batch['sequence'].to(device)
        }, {
            'density': targets
        })
        
        physics_loss = physics_losses['total_physics_loss']
        
        # Total loss
        total_batch_loss = recon_loss + 0.1 * chaos_loss + 0.05 * physics_loss
        
        # Backward pass
        total_batch_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Accumulate losses
        total_loss += total_batch_loss.item()
        total_recon_loss += recon_loss.item()
        total_physics_loss += physics_loss.item()
        total_chaos_loss += chaos_loss.item()
        
        # Log to tensorboard
        global_step = epoch * len(train_loader) + batch_idx
        if batch_idx % 50 == 0:
            writer.add_scalar('Train/Batch_Total_Loss', total_batch_loss.item(), global_step)
            writer.add_scalar('Train/Batch_Recon_Loss', recon_loss.item(), global_step)
            writer.add_scalar('Train/Batch_Physics_Loss', physics_loss.item(), global_step)
            writer.add_scalar('Train/Batch_Chaos_Loss', chaos_loss.item(), global_step)
        
        # Update progress bar description
        pbar.set_postfix({
            'loss': f'{total_batch_loss.item():.4f}',
            'recon': f'{recon_loss.item():.4f}',
            'phys': f'{physics_loss.item():.4f}'
        })
            
    # Average losses
    avg_loss = total_loss / len(train_loader)
    avg_recon_loss = total_recon_loss / len(train_loader)
    avg_physics_loss = total_physics_loss / len(train_loader)
    avg_chaos_loss = total_chaos_loss / len(train_loader)
    
    return {
        'total_loss': avg_loss,
        'recon_loss': avg_recon_loss,
        'physics_loss': avg_physics_loss,
        'chaos_loss': avg_chaos_loss
    }

def validate_epoch(model: nn.Module,
                  val_loader: DataLoader,
                  physics_regularizer: PhysicsRegularizer,
                  device: str) -> Dict[str, float]:
    """Validate one epoch"""
    model.eval()
    
    total_loss = 0.0
    total_recon_loss = 0.0
    total_physics_loss = 0.0
    total_chaos_loss = 0.0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation', leave=True)
        for batch in pbar:
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            chaos_targets = batch['chaos_features'].to(device)
            
            outputs = model(inputs)
            
            # Calculate losses
            recon_loss = F.mse_loss(outputs['reconstructed'], targets)
            chaos_loss = F.mse_loss(outputs['physics_features'], chaos_targets)
            
            physics_losses = physics_regularizer({
                'density': outputs['reconstructed'],
                'density_sequence': batch['sequence'].to(device)
            }, {
                'density': targets
            })
            
            physics_loss = physics_losses['total_physics_loss']
            total_batch_loss = recon_loss + 0.1 * chaos_loss + 0.05 * physics_loss
            
            total_loss += total_batch_loss.item()
            total_recon_loss += recon_loss.item()
            total_physics_loss += physics_loss.item()
            total_chaos_loss += chaos_loss.item()
            
            # Update progress bar description
            pbar.set_postfix({
                'loss': f'{total_batch_loss.item():.4f}',
                'recon': f'{recon_loss.item():.4f}'
            })
            
    return {
        'total_loss': total_loss / len(val_loader),
        'recon_loss': total_recon_loss / len(val_loader),
        'physics_loss': total_physics_loss / len(val_loader),
        'chaos_loss': total_chaos_loss / len(val_loader)
    }

def main():
    parser = argparse.ArgumentParser(description='SmokePhysAI Training')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Setup experiment
    exp_dir, writer, device = setup_experiment(config)
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        batch_size=config['training']['batch_size'],
        num_train=config['data']['num_train'],
        num_val=config['data']['num_val'],
        grid_size=tuple(config['data']['grid_size']),
        device=device,
        cache_dir=config['data']['cache_dir']
    )
    
    # Create model
    from src.models.smokephys_net import SmokePhysNet
    from src.models.physics_regularizer import PhysicsRegularizer
    
    model = SmokePhysNet(
        input_dim=config['model']['input_dim'],
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers'],
        num_heads=config['model']['num_heads'],
        chaos_strength=config['model']['chaos_strength']
    ).to(device)
    
    physics_regularizer = PhysicsRegularizer(
        conservation_weight=config['physics']['conservation_weight'],
        continuity_weight=config['physics']['continuity_weight'],
        energy_weight=config['physics']['energy_weight']
    )
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['num_epochs']
    )
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(config['training']['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['training']['num_epochs']}")
        
        # Train and validate
        train_metrics = train_epoch(
            model, train_loader, optimizer, physics_regularizer,
            device, epoch, writer
        )
        
        val_metrics = validate_epoch(
            model, val_loader, physics_regularizer, device
        )
        
        # Learning rate scheduling
        scheduler.step()
        
        # Log to tensorboard
        writer.add_scalar('Train/Epoch_Loss', train_metrics['total_loss'], epoch)
        writer.add_scalar('Val/Epoch_Loss', val_metrics['total_loss'], epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Print metrics
        print(f"\nEpoch Summary:")
        print(f"Train Loss: {train_metrics['total_loss']:.4f}")
        print(f"Val Loss: {val_metrics['total_loss']:.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_metrics['total_loss'] < best_val_loss:
            best_val_loss = val_metrics['total_loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_metrics['total_loss'],
                'config': config
            }, os.path.join(exp_dir, 'best_model.pth'))
            
    print("Training completed!")
    writer.close()

if __name__ == "__main__":
    main()
