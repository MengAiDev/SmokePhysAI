import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from typing import Dict, List, Tuple
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class RobustnessEvaluator:
    """Robustness evaluator"""
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        
    def evaluate_physics_consistency(self, 
                                   model: nn.Module,
                                   test_data: torch.Tensor,
                                   physics_targets: Dict) -> Dict:
        """Evaluate physics consistency"""
        model.eval()
        
        with torch.no_grad():
            predictions = model(test_data)
            
        # Calculate physics feature errors
        physics_pred = predictions['physics_features']
        
        metrics = {}
        
        if 'lyapunov' in physics_targets:
            lyapunov_error = torch.abs(
                physics_pred[:, 0] - physics_targets['lyapunov']
            ).mean()
            metrics['lyapunov_mae'] = lyapunov_error.item()
            
        if 'fractal_dimension' in physics_targets:
            fractal_error = torch.abs(
                physics_pred[:, 1] - physics_targets['fractal_dimension']
            ).mean()
            metrics['fractal_mae'] = fractal_error.item()
            
        if 'entropy' in physics_targets:
            entropy_error = torch.abs(
                physics_pred[:, 2] - physics_targets['entropy']
            ).mean()
            metrics['entropy_mae'] = entropy_error.item()
            
        return metrics
        
    def evaluate_reconstruction_quality(self,
                                      model: nn.Module,
                                      test_data: torch.Tensor,
                                      targets: torch.Tensor) -> Dict:
        """Evaluate reconstruction quality"""
        model.eval()
        
        with torch.no_grad():
            predictions = model(test_data)
            reconstructed = predictions['reconstructed']
            
        # SSIM
        ssim_score = self.compute_ssim(reconstructed, targets)
        
        # PSNR
        psnr_score = self.compute_psnr(reconstructed, targets)
        
        # MSE
        mse_score = F.mse_loss(reconstructed, targets).item()
        
        return {
            'ssim': ssim_score,
            'psnr': psnr_score,
            'mse': mse_score
        }
        
    def compute_ssim(self, pred: torch.Tensor, target: torch.Tensor, 
                     window_size: int = 11, sigma: float = 1.5) -> float:
        """Compute SSIM"""
        # Simplified SSIM implementation
        mu1 = F.avg_pool2d(pred, window_size, stride=1, padding=window_size//2)
        mu2 = F.avg_pool2d(target, window_size, stride=1, padding=window_size//2)
        
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.avg_pool2d(pred * pred, window_size, stride=1, padding=window_size//2) - mu1_sq
        sigma2_sq = F.avg_pool2d(target * target, window_size, stride=1, padding=window_size//2) - mu2_sq
        sigma12 = F.avg_pool2d(pred * target, window_size, stride=1, padding=window_size//2) - mu1_mu2
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return ssim_map.mean().item()
        
    def compute_psnr(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Compute PSNR"""
        mse = F.mse_loss(pred, target)
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
        return psnr.item()
