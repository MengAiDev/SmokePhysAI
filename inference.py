import torch
import argparse
import yaml
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# 导入自定义模块
from src.utils.data_loader import SyntheticSmokeDataset
from src.utils.visualization import SmokeVisualizer
from src.models.smokephys_net import SmokePhysNet
from src.physics.smoke_simulator import SmokeSimulator

def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_model(config: dict, checkpoint_path: str, device: str) -> SmokePhysNet:
    """加载预训练模型"""
    model = SmokePhysNet(
        input_dim=config['model']['input_dim'],
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers'],
        num_heads=config['model']['num_heads'],
        chaos_strength=config['model']['chaos_strength']
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def generate_test_sequence(simulator, sequence_length=20):
    """生成测试序列"""
    simulator.ns_solver.setup_grid()
    
    # 添加烟雾源
    positions = [(64, 64), (32, 32), (96, 96)]
    intensities = [1.5, 1.0, 0.8]
    simulator.add_incense_source(positions, intensities)
    
    # 生成序列
    sequence = []
    for _ in tqdm(range(sequence_length), desc="生成烟雾序列"):
        density = simulator.simulate_step()
        sequence.append(density.clone().cpu().numpy())
    
    return sequence

def run_inference(model, sequence, device):
    """运行模型推理"""
    predictions = []
    physics_features = []
    
    with torch.no_grad():
        for i in tqdm(range(len(sequence) - 1), desc="运行推理"):
            # 准备输入数据
            input_frame = torch.tensor(sequence[i], device=device).unsqueeze(0).unsqueeze(0).float()
            
            # 模型预测
            output = model(input_frame)
            
            # 保存结果
            reconstructed = output['reconstructed'].squeeze().cpu().numpy()
            predictions.append(reconstructed)
            
            # 保存物理特征
            phys_feat = output['physics_features'].squeeze().cpu().numpy()
            physics_features.append(phys_feat)
    
    return predictions, physics_features

def visualize_results(ground_truth, predictions, physics_features):
    """可视化结果"""
    visualizer = SmokeVisualizer(figsize=(15, 10))
    
    # 可视化烟雾演化
    combined_sequence = ground_truth[1:]  # 从第二帧开始作为目标
    visualizer.plot_smoke_evolution(combined_sequence, save_path="ground_truth.png")
    visualizer.plot_smoke_evolution(predictions, save_path="predictions.png")
    
    # 可视化物理特征
    chaos_metrics = {
        'lyapunov_exponent': [feat[0] for feat in physics_features],
        'fractal_dimension': [feat[1] for feat in physics_features],
        'entropy': [feat[2] for feat in physics_features]
    }
    visualizer.plot_chaos_features(chaos_metrics, save_path="physics_features.png")
    
    # 对比特定帧
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    frame_indices = [0, len(predictions)//2, -1]
    
    for i, idx in enumerate(frame_indices):
        # 真实帧
        axes[0, i].imshow(ground_truth[idx+1], cmap='hot')
        axes[0, i].set_title(f'真实帧 {idx+1}')
        axes[0, i].axis('off')
        
        # 预测帧
        axes[1, i].imshow(predictions[idx], cmap='hot')
        axes[1, i].set_title(f'预测帧 {idx+1}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig("comparison.png", dpi=150)
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='SmokePhysAI 推理脚本')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='模型检查点路径')
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载模型
    model = load_model(config, args.checkpoint, str(device))
    
    # 创建烟雾模拟器
    simulator = SmokeSimulator(
        grid_size=tuple(config['simulation']['grid_size']),
        dt=config['simulation']['dt'],
        viscosity=config['simulation']['viscosity'],
        device=str(device)
    )
    
    # 生成测试序列
    sequence = generate_test_sequence(simulator, sequence_length=20)
    
    # 运行推理
    predictions, physics_features = run_inference(model, sequence, device)
    
    # 可视化结果
    visualize_results(sequence, predictions, physics_features)
    print("可视化结果已保存到当前目录")

if __name__ == "__main__":
    main()
