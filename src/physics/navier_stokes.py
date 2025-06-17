import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional

class NavierStokesSimulator(nn.Module):
    """简化的Navier-Stokes方程求解器，用于烟雾动力学仿真"""
    
    def __init__(self, 
                 grid_size: Tuple[int, int] = (128, 128),
                 dt: float = 0.01,
                 viscosity: float = 0.001,
                 device: str = 'cuda'):
        super().__init__()
        self.grid_size = grid_size
        self.dt = dt
        self.viscosity = viscosity
        self.device = device
        
        # 初始化网格
        self.h, self.w = grid_size
        self.setup_grid()
        
    def setup_grid(self):
        """设置计算网格"""
        # 修改速度场尺寸确保维度匹配
        self.u = torch.zeros((self.h + 1, self.w), device=self.device)  # 改为 (h+1, w)
        self.v = torch.zeros((self.h, self.w + 1), device=self.device)  # 改为 (h, w+1)
        
        # 压力场和密度场
        self.p = torch.zeros(self.h, self.w, device=self.device)
        self.density = torch.zeros(self.h, self.w, device=self.device)
        
        # 边界条件标记
        self.boundary = torch.zeros(self.h, self.w, device=self.device)
        
    def add_smoke_source(self, x: int, y: int, radius: int = 10, intensity: float = 1.0):
        """添加烟雾源"""
        y_grid, x_grid = torch.meshgrid(
            torch.arange(self.h, device=self.device),
            torch.arange(self.w, device=self.device),
            indexing='ij'
        )
        
        dist = torch.sqrt((x_grid - x)**2 + (y_grid - y)**2)
        mask = dist <= radius
        
        self.density[mask] += intensity * torch.exp(-dist[mask]**2 / (2 * (radius/3)**2))
        
    def diffusion_step(self, field: torch.Tensor, viscosity: float) -> torch.Tensor:
        """扩散步骤"""
        # 创建带有边界的临时场
        padded = torch.zeros(field.shape[0] + 2, field.shape[1] + 2, device=self.device)
        padded[1:-1, 1:-1] = field
        
        # 复制边界值
        padded[0, 1:-1] = field[0]  # 上边界
        padded[-1, 1:-1] = field[-1]  # 下边界
        padded[1:-1, 0] = field[:, 0]  # 左边界
        padded[1:-1, -1] = field[:, -1]  # 右边界
        
        # 角点复制
        padded[0, 0] = field[0, 0]
        padded[0, -1] = field[0, -1]
        padded[-1, 0] = field[-1, 0]
        padded[-1, -1] = field[-1, -1]
        
        # 计算拉普拉斯算子
        laplacian = (padded[:-2, 1:-1] + padded[2:, 1:-1] + 
                    padded[1:-1, :-2] + padded[1:-1, 2:] - 4 * field)
        
        return field + self.dt * viscosity * laplacian
        
    def advection_step(self, field: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """平流步骤 - 使用反向欧拉法"""
        h, w = field.shape
        
        # 创建坐标网格
        y_coords = torch.arange(h, device=self.device, dtype=torch.float32)
        x_coords = torch.arange(w, device=self.device, dtype=torch.float32)
        Y, X = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # 计算反向追踪的坐标
        u_interp = self.interpolate_velocity_u(u, Y, X)
        v_interp = self.interpolate_velocity_v(v, Y, X)
        
        prev_x = X - self.dt * u_interp
        prev_y = Y - self.dt * v_interp
        
        # 边界处理
        prev_x = torch.clamp(prev_x, 0, w - 1)
        prev_y = torch.clamp(prev_y, 0, h - 1)
        
        # 双线性插值
        return self.bilinear_interpolate(field, prev_y, prev_x)
        
    def interpolate_velocity_u(self, u: torch.Tensor, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """插值u速度分量到密度网格"""
        # u在(i, j+0.5)位置，需要插值到(i, j)
        x_u = x + 0.5
        x_u = torch.clamp(x_u, 0, u.shape[1] - 1)
        return self.bilinear_interpolate(u, y, x_u)
        
    def interpolate_velocity_v(self, v: torch.Tensor, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """插值v速度分量到密度网格"""
        # v在(i+0.5, j)位置，需要插值到(i, j)
        y_v = y + 0.5
        y_v = torch.clamp(y_v, 0, v.shape[0] - 1)
        return self.bilinear_interpolate(v, y_v, x)
        
    def bilinear_interpolate(self, field: torch.Tensor, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """双线性插值"""
        h, w = field.shape
        
        x0 = torch.floor(x).long()
        x1 = x0 + 1
        y0 = torch.floor(y).long()
        y1 = y0 + 1
        
        x0 = torch.clamp(x0, 0, w - 1)
        x1 = torch.clamp(x1, 0, w - 1)
        y0 = torch.clamp(y0, 0, h - 1)
        y1 = torch.clamp(y1, 0, h - 1)
        
        wa = (x1.float() - x) * (y1.float() - y)
        wb = (x - x0.float()) * (y1.float() - y)
        wc = (x1.float() - x) * (y - y0.float())
        wd = (x - x0.float()) * (y - y0.float())
        
        return (wa * field[y0, x0] + wb * field[y0, x1] + 
                wc * field[y1, x0] + wd * field[y1, x1])
        
    def pressure_projection(self):
        """压力投影步骤 - 使用雅可比迭代"""
        # 计算速度散度
        div = (self.u[1:, :] - self.u[:-1, :] + self.v[:, 1:] - self.v[:, :-1]) / self.dt
        
        # 求解泊松方程
        for _ in range(20):  # 雅可比迭代
            p_new = torch.zeros_like(self.p)
            p_new[1:-1, 1:-1] = 0.25 * (
                self.p[:-2, 1:-1] + self.p[2:, 1:-1] + 
                self.p[1:-1, :-2] + self.p[1:-1, 2:] - div[1:-1, 1:-1]
            )
            self.p = p_new
            
        # 更新速度场
        self.u[1:-1, :] -= self.dt * (self.p[1:, :] - self.p[:-1, :])
        self.v[:, 1:-1] -= self.dt * (self.p[:, 1:] - self.p[:, :-1])
    
    def step(self) -> torch.Tensor:
        """执行一个时间步"""
        # 1. 添加外力 (浮力)
        buoyancy = self.density * 0.1
        self.v[:, :-1] += self.dt * buoyancy  # 修改前为：self.v[:-1, :] += self.dt * buoyancy
        
        # 2. 扩散
        self.u = self.diffusion_step(self.u, self.viscosity)
        self.v = self.diffusion_step(self.v, self.viscosity)
        self.density = self.diffusion_step(self.density, self.viscosity * 0.1)
        
        # 3. 压力投影
        self.pressure_projection()
        
        # 4. 平流
        self.u = self.advection_step(self.u, self.u, self.v)
        self.v = self.advection_step(self.v, self.u, self.v)
        self.density = self.advection_step(self.density, self.u, self.v)
        
        # 5. 密度衰减
        self.density *= 0.995
        
        return self.density.clone()