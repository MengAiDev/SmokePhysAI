# Physics robustness testing
# Implements evaluation of model's physical consistency

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from smoke_phys.simulator import SmokeSimulator
from smoke_phys.evaluator import PhysRobustnessEvaluator
from models.model import PhysicsAwareNet
from torch.utils.data import DataLoader

def test_phys_robustness(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PhysicsAwareNet().to(device)
    model.load_state_dict(torch.load(model_path))
    
    # 创建测试数据集
    from torchvision import datasets, transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    test_dataset = datasets.MNIST(
        root='./data', train=False, transform=transform, download=True
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 初始化评估器
    evaluator = PhysRobustnessEvaluator(model, device)
    
    # 评估湍流鲁棒性
    results = evaluator.evaluate_turbulence(test_loader)
    
    # 打印结果
    print("物理扰动鲁棒性评估结果：")
    for res in results:
        print(f"雷诺数 {res['reynolds']}: 损失 = {res['loss']:.4f}")

if __name__ == "__main__":
    test_phys_robustness('models/phys_aware.pth')