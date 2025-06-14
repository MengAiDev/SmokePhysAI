# Physics simulation engine
# Contains core smoke physics modeling and simulation logic

import torch
import torch.nn.functional as F
from contextlib import nullcontext

class SmokeSimulator:
    def __init__(self, resolution=256, device='cpu'):
        self.res = resolution
        self.device = device
        self.dt = 0.1
        self.viscosity = 0.01
        
        # 初始化物理场
        self.velocity = torch.zeros(2, resolution, resolution, device=device)
        self.density = torch.rand(resolution, resolution, device=device) * 0.1
        self.temperature = torch.zeros(resolution, resolution, device=device)
        
        # Lorenz系统参数
        self.lx, self.ly, self.lz = 0.1, 0.1, 0.1
        self.sigma, self.rho, self.beta = 10.0, 28.0, 8/3.0
        
        # 卷积核
        self.laplace_kernel = torch.tensor([
            [0.0, 1.0, 0.0],
            [1.0, -4.0, 1.0],
            [0.0, 1.0, 0.0]
        ], dtype=torch.float32, device=device).repeat(2, 1, 1, 1)
        
        # 创建网格坐标
        self.grid_x, self.grid_y = torch.meshgrid(
            torch.linspace(-1, 1, resolution, device=device),
            torch.linspace(-1, 1, resolution, device=device)
        )
        # Precompute reusable tensors
        self.grid = torch.stack([self.grid_x, self.grid_y], dim=-1).unsqueeze(0)
        
        # 预分配CUDA内存以减少动态分配
        self.buffer1 = torch.zeros(2, resolution, resolution, device=device)
        self.buffer2 = torch.zeros(2, resolution, resolution, device=device)
        
        # 使用固定内存加速CPU到GPU传输
        self.cpu_buffer = torch.zeros(2, resolution, resolution).pin_memory()
        
        # CUDA Stream for async operations
        self.stream = torch.cuda.Stream() if device == 'cuda' else None
        
    def add_heat_source(self, position, intensity):
        """添加热源"""
        x = int(position[0] * self.res)
        y = int(position[1] * self.res)
        radius = max(2, int(intensity * 10))
        
        for i in range(-radius, radius+1):
            for j in range(-radius, radius+1):
                if 0 <= x+i < self.res and 0 <= y+j < self.res:
                    dist = torch.sqrt(torch.tensor(i**2 + j**2))
                    if dist <= radius:
                        self.temperature[y+j, x+i] = intensity * (1 - dist/radius)

    def lorenz_step(self):
        """Lorenz系统演化"""
        dx = self.sigma * (self.ly - self.lx)
        dy = self.lx * (self.rho - self.lz) - self.ly
        dz = self.lx * self.ly - self.beta * self.lz
        
        self.lx += self.dt * dx
        self.ly += self.dt * dy
        self.lz += self.dt * dz

    def diffuse_velocity(self):
        """速度场扩散"""
        # Use grouped convolution for efficiency
        laplacian = F.conv2d(self.velocity.unsqueeze(0), 
                           self.laplace_kernel, padding=1, groups=2)
        self.velocity.add_(self.dt * self.viscosity * laplacian.squeeze(0))
        
    def advect_velocity(self):
        """速度场对流"""
        displacement = torch.stack([
            self.velocity[0] * self.dt * self.res / 10,
            self.velocity[1] * self.dt * self.res / 10
        ]).permute(1,2,0).unsqueeze(0)
        
        sampled = F.grid_sample(
            self.velocity.unsqueeze(0), self.grid - displacement,
            mode='bilinear', padding_mode='border', align_corners=False
        )
        self.velocity = sampled.squeeze(0)
        
    def compute_fractal_dim(self):
        """GPU-optimized fractal dimension计算"""
        threshold = 0.5
        binary = (self.density > threshold).float()
        
        sizes = [4, 8, 16, 32, 64]  # Powers of 2
        counts = []
        
        for size in sizes:
            pool = F.adaptive_max_pool2d(binary.unsqueeze(0).unsqueeze(0), (size, size))
            counts.append(pool.sum())
            
        log_sizes = torch.log(torch.tensor(sizes, device=self.device))
        log_counts = torch.log(torch.stack(counts).squeeze())
        slope = torch.polyfit(log_sizes, log_counts, 1)[0]
        return -slope.item()

    def step(self):
        """单步模拟优化版本"""
        with torch.cuda.stream(self.stream) if self.stream else nullcontext():
            # 使用预分配的buffer
            torch.cuda.synchronize()  # 确保之前的计算完成
            
            self.lorenz_step()
            
            # 使用融合操作减少内存访问
            self._fused_diffuse_advect()
            
            # 异步更新温度场
            self._async_temperature_update()
            
    def _compute_displacement(self):
        """计算位移场"""
        displacement = torch.stack([
            self.velocity[0] * self.dt * self.res / 10,
            self.velocity[1] * self.dt * self.res / 10
        ], dim=-1).permute(1, 2, 0).unsqueeze(0)
        
        return self.grid - displacement

    def _fused_diffuse_advect(self):
        """融合扩散和对流计算"""
        with torch.cuda.amp.autocast():
            # 使用预分配buffer减少内存分配
            self.buffer1.copy_(self.velocity)
            
            # Fused operation
            laplacian = F.conv2d(self.buffer1.unsqueeze(0), 
                               self.laplace_kernel, padding=1, groups=2)
            self.velocity.add_(self.dt * self.viscosity * laplacian.squeeze(0))
            
            # 优化对流计算
            displacement = self._compute_displacement()
            self.velocity.copy_(F.grid_sample(
                self.buffer1.unsqueeze(0),
                displacement,
                mode='bilinear',
                padding_mode='border',
                align_corners=False
            ).squeeze(0))
            
    def _async_temperature_update(self):
        """异步温度场更新"""
        if self.stream:
            with torch.cuda.stream(self.stream):
                self.temperature.mul_(0.99)
                self.density.addcmul_(self.temperature, self.dt)
                self.density.clamp_(0, 1)

    def run_steps(self, steps=100):
        """运行多步模拟"""
        for _ in range(steps):
            self.step()
        return self.density.clone()