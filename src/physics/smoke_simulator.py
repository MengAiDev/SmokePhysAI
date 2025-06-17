import torch
import torch.nn as nn
import numpy as np

from .navier_stokes import NavierStokesSimulator
from .fractal_generator import FractalGenerator

class SmokeSimulator(nn.Module):
    """Complete smoke physics simulation system"""
    
    def __init__(self,
                 grid_size: tuple = (128, 128),
                 dt: float = 0.01,
                 viscosity: float = 0.001,
                 device: str = 'cuda'):
        super().__init__()
        
        self.ns_solver = NavierStokesSimulator(grid_size, dt, viscosity, device)
        self.fractal_gen = FractalGenerator(device)
        self.device = device
        
        # Smoke history record
        self.history = []
        self.max_history = 100
        
    def add_incense_source(self, positions: list, intensities: list):
        """Add incense smoke source"""
        for (x, y), intensity in zip(positions, intensities):
            self.ns_solver.add_smoke_source(x, y, radius=8, intensity=intensity)
            
    def simulate_step(self, add_fractal: bool = True) -> torch.Tensor:
        """Execute one simulation step"""
        # Basic physical simulation
        density = self.ns_solver.step()
        
        # Add fractal characteristics
        if add_fractal:
            density = self.fractal_gen.apply_fractal_perturbation(density, intensity=0.05)
            
        # Record history
        self.history.append(density.clone())
        if len(self.history) > self.max_history:
            self.history.pop(0)
            
        return density
        
    def get_chaos_features(self) -> dict:
        """Calculate chaos features"""
        if len(self.history) < 10:
            return {}
            
        # Calculate Lyapunov exponent
        lyapunov = self.compute_lyapunov_exponent()
        
        # Calculate fractal dimension
        fractal_dim = self.compute_fractal_dimension()
        
        # Calculate entropy
        entropy = self.compute_entropy()
        
        return {
            'lyapunov_exponent': lyapunov,
            'fractal_dimension': fractal_dim,
            'entropy': entropy
        }
        
    def compute_lyapunov_exponent(self) -> float:
        """Calculate Lyapunov exponent"""
        if len(self.history) < 20:
            return 0.0
            
        # Simplified Lyapunov exponent calculation
        states = torch.stack(self.history[-20:])
        
        # Calculate distances between adjacent states
        distances = []
        for i in range(len(states) - 1):
            dist = torch.norm(states[i+1] - states[i])
            distances.append(dist.item())
            
        # Calculate average divergence rate
        if len(distances) > 1:
            log_distances = np.log(np.array(distances) + 1e-8)
            lyapunov = np.mean(np.diff(log_distances))
            return max(0, lyapunov) # type: ignore
        
        return 0.0
        
    def compute_fractal_dimension(self) -> float:
        """Calculate fractal dimension (box-counting method)"""
        if not self.history:
            return 0.0
            
        current_state = self.history[-1]
        
        # Binarization
        binary = (current_state > current_state.mean()).float()
        
        # Box counting at different scales
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
            
        # Calculate fractal dimension
        if len(counts) > 1:
            log_scales = np.log(scales)
            log_counts = np.log(np.array(counts) + 1)
            slope = np.polyfit(log_scales, log_counts, 1)[0]
            return abs(slope)
            
        return 1.0
        
    def compute_entropy(self) -> float:
        """Calculate information entropy"""
        if not self.history:
            return 0.0
            
        current_state = self.history[-1]
        
        # Discretization
        current_state_cpu = current_state.detach().cpu()  # Ensure on CPU
        hist = torch.histogram(current_state_cpu.flatten(), bins=256, range=(0, 1))
        probs = hist.hist.float() / hist.hist.sum()
        
        # Calculate entropy
        entropy = -torch.sum(probs * torch.log2(probs + 1e-8))
        return entropy.item()