import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional

class NavierStokesSimulator(nn.Module):
    """Simplified Navier-Stokes equation solver for smoke dynamics simulation"""
    
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
        
        # Initialize grid
        self.h, self.w = grid_size
        self.setup_grid()
        
    def setup_grid(self):
        """Setup computational grid"""
        # Modify velocity field dimensions to ensure dimension matching
        self.u = torch.zeros((self.h + 1, self.w), device=self.device)  # Change to (h+1, w)
        self.v = torch.zeros((self.h, self.w + 1), device=self.device)  # Change to (h, w+1)
        
        # Pressure and density fields
        self.p = torch.zeros(self.h, self.w, device=self.device)
        self.density = torch.zeros(self.h, self.w, device=self.device)
        
        # Boundary condition markers
        self.boundary = torch.zeros(self.h, self.w, device=self.device)
        
    def add_smoke_source(self, x: int, y: int, radius: int = 10, intensity: float = 1.0):
        """Add smoke source"""
        y_grid, x_grid = torch.meshgrid(
            torch.arange(self.h, device=self.device),
            torch.arange(self.w, device=self.device),
            indexing='ij'
        )
        
        dist = torch.sqrt((x_grid - x)**2 + (y_grid - y)**2)
        mask = dist <= radius
        
        self.density[mask] += intensity * torch.exp(-dist[mask]**2 / (2 * (radius/3)**2))
        
    def diffusion_step(self, field: torch.Tensor, viscosity: float) -> torch.Tensor:
        """Diffusion step"""
        # Create temporary field with boundaries
        padded = torch.zeros(field.shape[0] + 2, field.shape[1] + 2, device=self.device)
        padded[1:-1, 1:-1] = field
        
        # Copy boundary values
        padded[0, 1:-1] = field[0]  # Top boundary
        padded[-1, 1:-1] = field[-1]  # Bottom boundary
        padded[1:-1, 0] = field[:, 0]  # Left boundary
        padded[1:-1, -1] = field[:, -1]  # Right boundary
        
        # Copy corner points
        padded[0, 0] = field[0, 0]
        padded[0, -1] = field[0, -1]
        padded[-1, 0] = field[-1, 0]
        padded[-1, -1] = field[-1, -1]
        
        # Calculate Laplacian operator
        laplacian = (padded[:-2, 1:-1] + padded[2:, 1:-1] + 
                    padded[1:-1, :-2] + padded[1:-1, 2:] - 4 * field)
        
        return field + self.dt * viscosity * laplacian
        
    def advection_step(self, field: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Advection step - using backward Euler method"""
        h, w = field.shape
        
        # Create coordinate grid
        y_coords = torch.arange(h, device=self.device, dtype=torch.float32)
        x_coords = torch.arange(w, device=self.device, dtype=torch.float32)
        Y, X = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Calculate back-traced coordinates
        u_interp = self.interpolate_velocity_u(u, Y, X)
        v_interp = self.interpolate_velocity_v(v, Y, X)
        
        prev_x = X - self.dt * u_interp
        prev_y = Y - self.dt * v_interp
        
        # Boundary handling
        prev_x = torch.clamp(prev_x, 0, w - 1)
        prev_y = torch.clamp(prev_y, 0, h - 1)
        
        # Bilinear interpolation
        return self.bilinear_interpolate(field, prev_y, prev_x)
        
    def interpolate_velocity_u(self, u: torch.Tensor, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Interpolate u velocity component to density grid"""
        # u at position (i, j+0.5), needs interpolation to (i, j)
        x_u = x + 0.5
        x_u = torch.clamp(x_u, 0, u.shape[1] - 1)
        return self.bilinear_interpolate(u, y, x_u)
        
    def interpolate_velocity_v(self, v: torch.Tensor, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Interpolate v velocity component to density grid"""
        # v at position (i+0.5, j), needs interpolation to (i, j)
        y_v = y + 0.5
        y_v = torch.clamp(y_v, 0, v.shape[0] - 1)
        return self.bilinear_interpolate(v, y_v, x)
        
    def bilinear_interpolate(self, field: torch.Tensor, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Bilinear interpolation"""
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
        """Pressure projection step - using Jacobi iteration"""
        # Calculate velocity divergence
        div = (self.u[1:, :] - self.u[:-1, :] + self.v[:, 1:] - self.v[:, :-1]) / self.dt
        
        # Solve Poisson equation
        for _ in range(20):  # Jacobi iteration
            p_new = torch.zeros_like(self.p)
            p_new[1:-1, 1:-1] = 0.25 * (
                self.p[:-2, 1:-1] + self.p[2:, 1:-1] + 
                self.p[1:-1, :-2] + self.p[1:-1, 2:] - div[1:-1, 1:-1]
            )
            self.p = p_new
            
        # Update velocity field
        self.u[1:-1, :] -= self.dt * (self.p[1:, :] - self.p[:-1, :])
        self.v[:, 1:-1] -= self.dt * (self.p[:, 1:] - self.p[:, :-1])
    
    def step(self) -> torch.Tensor:
        """Execute one time step"""
        # 1. Add external forces (buoyancy)
        buoyancy = self.density * 0.1
        self.v[:, :-1] += self.dt * buoyancy  # Changed from: self.v[:-1, :] += self.dt * buoyancy
        
        # 2. Diffusion
        self.u = self.diffusion_step(self.u, self.viscosity)
        self.v = self.diffusion_step(self.v, self.viscosity)
        self.density = self.diffusion_step(self.density, self.viscosity * 0.1)
        
        # 3. Pressure projection
        self.pressure_projection()
        
        # 4. Advection
        self.u = self.advection_step(self.u, self.u, self.v)
        self.v = self.advection_step(self.v, self.u, self.v)
        self.density = self.advection_step(self.density, self.u, self.v)
        
        # 5. Density decay
        self.density *= 0.995
        
        return self.density.clone()