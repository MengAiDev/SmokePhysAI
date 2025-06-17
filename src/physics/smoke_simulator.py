import torch
import torch.nn as nn
import numpy as np

from .navier_stokes import NavierStokesSimulator
from .fractal_generator import FractalGenerator

class SmokeSimulator(nn.Module):
    """完整的烟雾物理仿真系统"""
    
    def __init__(self,
                 grid_size: tuple = (128, 128),
                 dt: float = 0.01,
                 viscosity: float = 0.001,
                 device: str = 'cuda'):
        super().__init__()
        
        self.ns_solver = NavierStokesSimulator(grid_size, dt, viscosity, device)
        self.fractal_gen = FractalGenerator(device)
        self.device = device
        
        # 烟雾历史记录
        self.history = []
        self.max_history = 100
        
    def add_incense_source(self, positions: list, intensities: list):
        """添加香薰烟雾源"""
        for (x, y), intensity in zip(positions, intensities):
            self.ns_solver.add_smoke_source(x, y, radius=8, intensity=intensity)
            
    def simulate_step(self, add_fractal: bool = True) -> torch.Tensor:
        """执行一个仿真步骤"""
        # 基础物理仿真
        density = self.ns_solver.step()
        
        # 添加分形特性
        if add_fractal:
            density = self.fractal_gen.apply_fractal_perturbation(density, intensity=0.05)
            
        # 记录历史
        self.history.append(density.clone())
        if len(self.history) > self.max_history:
            self.history.pop(0)
            
        return density
        
    def get_chaos_features(self) -> dict:
        """计算混沌特征"""
        if len(self.history) < 10:
            return {}
            
        # 计算李雅普诺夫指数
        lyapunov = self.compute_lyapunov_exponent()
        
        # 计算分形维数
        fractal_dim = self.compute_fractal_dimension()
        
        # 计算熵
        entropy = self.compute_entropy()
        
        return {
            'lyapunov_exponent': lyapunov,
            'fractal_dimension': fractal_dim,
            'entropy': entropy
        }
        
    def compute_lyapunov_exponent(self) -> float:
        """计算李雅普诺夫指数"""
        if len(self.history) < 20:
            return 0.0
            
        # 简化的李雅普诺夫指数计算
        states = torch.stack(self.history[-20:])
        
        # 计算相邻状态间的距离
        distances = []
        for i in range(len(states) - 1):
            dist = torch.norm(states[i+1] - states[i])
            distances.append(dist.item())
            
        # 计算平均发散率
        if len(distances) > 1:
            log_distances = np.log(np.array(distances) + 1e-8)
            lyapunov = np.mean(np.diff(log_distances))
            return max(0, lyapunov)
        
        return 0.0
        
    def compute_fractal_dimension(self) -> float:
        """计算分形维数 (box-counting method)"""
        if not self.history:
            return 0.0
            
        current_state = self.history[-1]
        
        # 二值化
        binary = (current_state > current_state.mean()).float()
        
        # 不同尺度的box counting
        scales = [2, 4, 8, 16, 32]
        counts = []
        
        for scale in scales:
            h, w = binary.shape
            boxes_h = h // scale
            boxes_w = w // scale
            
            count = 0
            for i in range(boxes_h):
                for j in range(boxes_w):
                    box = binary[i*scale:(i+1)*scale, j*scale:(j+1)*scale]
                    if box.sum() > 0:
                        count += 1
                        
            counts.append(count)
            
        # 计算分形维数
        if len(counts) > 1:
            log_scales = np.log(scales)
            log_counts = np.log(np.array(counts) + 1)
            slope = np.polyfit(log_scales, log_counts, 1)[0]
            return abs(slope)
            
        return 1.0
        
    def compute_entropy(self) -> float:
        """计算信息熵"""
        if not self.history:
            return 0.0
            
        current_state = self.history[-1]
        
        # 离散化
        current_state_cpu = current_state.detach().cpu()  # 确保在CPU上
        hist = torch.histogram(current_state_cpu.flatten(), bins=256, range=(0, 1))
        probs = hist.hist.float() / hist.hist.sum()
        
        # 计算熵
        entropy = -torch.sum(probs * torch.log2(probs + 1e-8))
        return entropy.item()