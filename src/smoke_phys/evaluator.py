import torch
import torch.nn as nn
from smoke_phys.simulator import SmokeSimulator
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F

class PhysRobustnessEvaluator:
    """
    物理扰动鲁棒性评估器
    
    功能：
    1. 在不同物理参数下评估模型性能
    2. 支持烟雾扰动强度渐变测试
    3. 计算分形维度与模型性能的关系
    """
    
    def __init__(self, model, device='cpu'):
        """
        初始化评估器
        
        参数:
            model: 待评估的PyTorch模型
            device: 计算设备 ('cpu' 或 'cuda')
        """
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.model.eval()

    def evaluate_smoke_perturbation(self, dataloader, smoke_levels=[0.1, 0.3, 0.5, 0.7, 1.0]):
        """
        评估烟雾扰动下的模型性能
        
        参数:
            dataloader: 测试数据加载器
            smoke_levels: 烟雾强度级别列表
            
        返回:
            results: 包含每个级别准确率和损失的字典
        """
        results = {
            'smoke_levels': smoke_levels,
            'accuracies': [],
            'losses': []
        }
        
        for level in tqdm(smoke_levels, desc="烟雾强度测试"):
            total_correct = 0
            total_loss = 0
            total_samples = 0
            
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # 生成烟雾扰动
                perturbed_images = self._apply_smoke_perturbation(images, level)
                
                # 模型推理
                with torch.no_grad():
                    outputs, _ = self.model(perturbed_images)
                    loss = self.criterion(outputs, labels)
                
                # 统计结果
                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted == labels).sum().item()
                total_loss += loss.item() * images.size(0)
                total_samples += images.size(0)
            
            # 计算指标
            accuracy = total_correct / total_samples
            avg_loss = total_loss / total_samples
            results['accuracies'].append(accuracy)
            results['losses'].append(avg_loss)
            
            print(f"烟雾强度 {level:.1f} | 准确率: {accuracy*100:.2f}% | 损失: {avg_loss:.4f}")
        
        return results

    def evaluate_turbulence(self, dataloader, reynolds_list=[100, 500, 1000, 5000, 10000]):
        """
        评估湍流扰动下的模型性能
        
        参数:
            dataloader: 测试数据加载器
            reynolds_list: 雷诺数列表
            
        返回:
            results: 包含每个雷诺数下性能的字典
        """
        results = {
            'reynolds': reynolds_list,
            'accuracies': [],
            'losses': [],
            'fractal_dims': []
        }
        
        for reynolds in tqdm(reynolds_list, desc="湍流测试"):
            # 创建对应雷诺数的模拟器
            sim = SmokeSimulator(resolution=28, reynolds=reynolds, device=self.device)
            
            total_correct = 0
            total_loss = 0
            total_samples = 0
            total_fractal_dim = 0
            
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # 生成湍流扰动
                perturbed_images, fractal_dim = self._apply_turbulence_perturbation(images, sim)
                total_fractal_dim += fractal_dim
                
                # 模型推理
                with torch.no_grad():
                    outputs, _ = self.model(perturbed_images)
                    loss = self.criterion(outputs, labels)
                
                # 统计结果
                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted == labels).sum().item()
                total_loss += loss.item() * images.size(0)
                total_samples += images.size(0)
            
            # 计算指标
            accuracy = total_correct / total_samples
            avg_loss = total_loss / total_samples
            avg_fractal_dim = total_fractal_dim / total_samples
            
            results['accuracies'].append(accuracy)
            results['losses'].append(avg_loss)
            results['fractal_dims'].append(avg_fractal_dim)
            
            print(f"雷诺数 {reynolds} | 准确率: {accuracy*100:.2f}% | 损失: {avg_loss:.4f} | 分形维度: {avg_fractal_dim:.3f}")
        
        return results

    def _apply_smoke_perturbation(self, images, intensity):
        """
        应用烟雾扰动
        
        参数:
            images: 输入图像张量
            intensity: 烟雾强度 (0-1)
            
        返回:
            perturbed: 受扰动的图像
        """
        batch_size, _, h, w = images.shape
        
        # 创建烟雾模拟器
        sim = SmokeSimulator(resolution=h, device=self.device)
        sim.add_heat_source((0.5, 0.5), intensity)
        smoke_tex = sim.run_steps(int(50 * intensity))
        
        # 计算分形参数
        alpha = self._calculate_fractal_dim(smoke_tex)
        
        # 应用扰动
        smoke = smoke_tex.expand(batch_size, -1, -1)
        transmittance = torch.exp(-alpha * smoke).unsqueeze(1)
        scattered_light = 0.8 * alpha * smoke.unsqueeze(1)
        
        perturbed = images * transmittance + scattered_light
        return perturbed

    def _apply_turbulence_perturbation(self, images, sim):
        """
        应用湍流扰动
        
        参数:
            images: 输入图像张量
            sim: 配置好的烟雾模拟器
            
        返回:
            perturbed: 受扰动的图像
            fractal_dim: 分形维度
        """
        batch_size, _, h, w = images.shape
        
        # 运行模拟
        smoke_tex = sim.run_steps(50)
        fractal_dim = sim.compute_fractal_dim()
        
        # 计算分形参数
        alpha = fractal_dim / 3.0  # 归一化到 [0,1]
        
        # 应用扰动
        smoke = smoke_tex.expand(batch_size, -1, -1)
        transmittance = torch.exp(-alpha * smoke).unsqueeze(1)
        scattered_light = 0.8 * alpha * smoke.unsqueeze(1)
        
        perturbed = images * transmittance + scattered_light
        return perturbed, fractal_dim

    def _calculate_fractal_dim(self, density_field):
        """计算分形维度（内部方法）"""
        threshold = 0.5
        binary = (density_field > threshold).float()
        
        sizes = 2**np.arange(2, 8)
        counts = []
        
        for size in sizes:
            pool = F.adaptive_max_pool2d(binary.unsqueeze(0).unsqueeze(0), (size, size))
            counts.append(pool.sum().item())
            
        log_sizes = np.log(sizes)
        log_counts = np.log(counts)
        slope, _ = np.polyfit(log_sizes, log_counts, 1)
        return -slope