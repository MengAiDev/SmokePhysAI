# Model training script
# Implements physics-aware model training pipeline


import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import os
from models.model import PhysicsAwareNet
from smoke_phys.augmenter import PhysicsAugmenter
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import time

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # 加载基础数据集
    base_dataset = datasets.MNIST(
        root='./data', train=True, transform=transform, download=True
    )
    
    # 创建物理增强数据集
    train_dataset = PhysicsAugmenter(base_dataset, resolution=128, device=device)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # 初始化模型
    model = PhysicsAwareNet(num_classes=10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Print training configuration
    print("\n=== Training Configuration ===")
    print(f"Batch Size: 64")
    print(f"Learning Rate: 1e-3")
    print(f"Total Epochs: 10")
    print(f"Dataset Size: {len(train_dataset)}")
    print("===========================\n")
    
    best_acc = 0.0
    checkpoint_dir = "models/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    start_time = time.time()
    for epoch in range(10):
        epoch_start = time.time()
        model.train()
        total_loss = 0
        
        # Create progress bar
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/10')
        
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images, labels = images.to(device), labels.to(device)
            
            # 前向传播
            outputs, reg = model(images)
            task_loss = F.cross_entropy(outputs, labels.long())
            loss = task_loss + 0.5 * reg
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Update progress bar info
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Task Loss': f'{task_loss.item():.4f}',
                'Reg Loss': f'{reg.item():.4f}'
            })
        
        epoch_time = time.time() - epoch_start
        val_acc = validate_model(model, device)
        print(f"\n[Epoch {epoch+1} Statistics]")
        print(f"Average Loss: {total_loss/len(train_loader):.4f}")
        print(f"Validation Accuracy: {val_acc:.2%}")
        print(f"Time Elapsed: {epoch_time:.2f}s")
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"{checkpoint_dir}/best_model.pth")
            print(f"[SAVE] New best model saved (Accuracy: {best_acc:.2%})")
        
        # 定期保存checkpoint
        if (epoch + 1) % 2 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss/len(train_loader),
                'accuracy': val_acc
            }, f"{checkpoint_dir}/checkpoint_epoch_{epoch+1}.pth")
            print(f"[SAVE] Checkpoint saved for epoch {epoch+1}")
    
    total_time = time.time() - start_time
    print("\n=== Training Complete ===")
    print(f"Total Training Time: {total_time/60:.2f} minutes")
    print(f"Best Accuracy: {best_acc:.2%}")
    print(f"Final model saved to: models/phys_aware.pth")
    
    return model

def validate_model(model, device):
    print("\nStarting validation...")
    model.eval()
    correct = 0
    total = 0
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    val_dataset = datasets.MNIST(
        root='./data', train=False, transform=transform, download=True
    )
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs, _ = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return correct / total

if __name__ == "__main__":
    print("\n=== Starting Physics-Aware Model Training ===")
    train_model()
    print("=== Training Finished ===")