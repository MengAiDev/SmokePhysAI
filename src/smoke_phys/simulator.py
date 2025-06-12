# Physics simulation engine
# Contains core smoke physics modeling and simulation logic

import torch
import torch.nn.functional as F

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
        laplacian = F.conv2d(self.velocity.unsqueeze(0).float(), 
                           self.laplace_kernel.float(), padding=1, groups=2)
        self.velocity += self.dt * self.viscosity * laplacian.squeeze(0)

    def advect_velocity(self):
        """速度场对流"""
        displacement = torch.stack([
            self.velocity[0] * self.dt * self.res / 10,
            self.velocity[1] * self.dt * self.res / 10
        ]).permute(1,2,0).unsqueeze(0)
        
        grid = torch.stack([self.grid_x, self.grid_y], dim=-1).unsqueeze(0)
        sampled = F.grid_sample(
            self.velocity.unsqueeze(0), grid - displacement,
            mode='bilinear', padding_mode='border'
        )
        self.velocity = sampled.squeeze(0)

    def compute_fractal_dim(self):
        """实时分形维度计算"""
        threshold = 0.5
        binary = (self.density > threshold).float()
        
        sizes = 2**np.arange(2, 8)
        counts = []
        
        for size in sizes:
            pool = F.adaptive_max_pool2d(binary.unsqueeze(0).unsqueeze(0), 
                                       (size, size))
            counts.append(pool.sum().item())
            
        log_sizes = np.log(sizes)
        log_counts = np.log(counts)
        slope, _ = np.polyfit(log_sizes, log_counts, 1)
        return -slope

    def step(self):
        """单步模拟"""
        self.lorenz_step()
        self.diffuse_velocity()
        self.advect_velocity()
        
        # 添加混沌扰动
        perturb = 0.05 * torch.tensor([self.lx, self.ly], 
                                    device=self.device).view(2,1,1)
        self.velocity += perturb
        
        # 更新温度场
        self.temperature *= 0.99
        self.density = torch.clamp(self.density + self.temperature * self.dt, 0, 1)

    def run_steps(self, steps=100):
        """运行多步模拟"""
        for _ in range(steps):
            self.step()
        return self.density.clone()