import torch
import torch.nn as nn
import numpy as np

class FractalGenerator(nn.Module):
    """分形几何生成器，用于增强烟雾的分形特性"""
    
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        
    def generate_perlin_noise(self, shape: tuple, scale: float = 10.0) -> torch.Tensor:
        """生成Perlin噪声"""
        h, w = shape
        
        # 生成网格
        x = torch.linspace(0, scale, w, device=self.device)
        y = torch.linspace(0, scale, h, device=self.device)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        
        # 多层次噪声
        noise = torch.zeros_like(X)
        amplitude = 1.0
        frequency = 1.0
        
        for _ in range(6):  # 6个八度
            noise += amplitude * torch.sin(frequency * X) * torch.cos(frequency * Y)
            amplitude *= 0.5
            frequency *= 2.0
            
        return (noise + 1.0) / 2.0  # 归一化到 [0, 1]
        
    def generate_mandelbrot_field(self, shape: tuple, iterations: int = 100) -> torch.Tensor:
        """生成Mandelbrot集合作为分形场"""
        h, w = shape
        
        # 创建复数网格
        x = torch.linspace(-2.5, 1.5, w, device=self.device)
        y = torch.linspace(-1.5, 1.5, h, device=self.device)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        c = X + 1j * Y
        
        z = torch.zeros_like(c)
        escape_count = torch.zeros(h, w, device=self.device)
        
        for i in range(iterations):
            mask = torch.abs(z) <= 2
            z[mask] = z[mask]**2 + c[mask]
            escape_count[mask] = i
            
        return escape_count / iterations
        
    def apply_fractal_perturbation(self, field: torch.Tensor, intensity: float = 0.1) -> torch.Tensor:
        """应用分形扰动到场"""
        perlin = self.generate_perlin_noise(field.shape[-2:])
        mandelbrot = self.generate_mandelbrot_field(field.shape[-2:])
        
        # 组合分形场
        fractal_field = 0.7 * perlin + 0.3 * mandelbrot
        
        # 应用扰动
        return field + intensity * fractal_field * field