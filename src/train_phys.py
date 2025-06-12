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

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
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
    
    # 训练循环
    best_acc = 0.0
    checkpoint_dir = "models/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    for epoch in range(10):
        model.train()
        total_loss = 0
        
        for images, labels in train_loader:
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
        
        # 验证
        val_acc = validate_model(model, device)
        print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f} | Acc: {val_acc:.2%}")
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"{checkpoint_dir}/best_model.pth")
            print(f"New best model saved with accuracy: {best_acc:.2%}")
        
        # 定期保存checkpoint
        if (epoch + 1) % 2 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss/len(train_loader),
                'accuracy': val_acc
            }, f"{checkpoint_dir}/checkpoint_epoch_{epoch+1}.pth")
            print(f"Checkpoint saved at epoch {epoch+1}")
    
    # 保存最终模型
    torch.save(model.state_dict(), "models/phys_aware.pth")
    return model

def validate_model(model, device):
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
    train_model()