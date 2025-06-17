import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
from typing import List, Dict, Optional

class SmokeVisualizer:
    """烟雾可视化工具"""
    
    def __init__(self, figsize: tuple = (12, 8)):
        self.figsize = figsize
        plt.style.use('dark_background')
        
    def plot_smoke_evolution(self, 
                            density_sequence: List[torch.Tensor],
                            save_path: Optional[str] = None):
        """绘制烟雾演化过程"""
        num_frames = len(density_sequence)
        cols = min(8, num_frames)
        rows = (num_frames + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
        if rows == 1:
            axes = axes.reshape(1, -1)
            
        for i, density in enumerate(density_sequence):
            row, col = divmod(i, cols)
            ax = axes[row, col] if rows > 1 else axes[col]
            
            # 转换为numpy
            if isinstance(density, torch.Tensor):
                density_np = density.detach().cpu().numpy()
            else:
                density_np = density
                
            im = ax.imshow(density_np, cmap='hot', interpolation='bilinear')
            ax.set_title(f'Frame {i}')
            ax.axis('off')
            
        # 隐藏空白子图
        for i in range(num_frames, rows * cols):
            row, col = divmod(i, cols)
            ax = axes[row, col] if rows > 1 else axes[col]
            ax.axis('off')
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
    def plot_chaos_features(self, 
                           chaos_metrics: Dict[str, List[float]],
                           save_path: Optional[str] = None):
        """绘制混沌特征"""
        fig, axes = plt.subplots(1, 3, figsize=self.figsize)
        
        metrics = ['lyapunov_exponent', 'fractal_dimension', 'entropy']
        titles = ['Lyapunov Exponent', 'Fractal Dimension', 'Entropy']
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            if metric in chaos_metrics:
                axes[i].plot(chaos_metrics[metric], 'o-', linewidth=2, markersize=4)
                axes[i].set_title(title)
                axes[i].set_xlabel('Time Step')
                axes[i].grid(True, alpha=0.3)
                
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
    def plot_attention_maps(self,
                          attention_weights: torch.Tensor,
                          input_image: torch.Tensor,
                          save_path: Optional[str] = None):
        """可视化注意力权重"""
        # 取第一个头的注意力权重
        attn = attention_weights[0, 0].detach().cpu().numpy()  # [seq_len, seq_len]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 原始图像
        if isinstance(input_image, torch.Tensor):
            img = input_image[0, 0].detach().cpu().numpy()
        else:
            img = input_image
            
        axes[0].imshow(img, cmap='hot')
        axes[0].set_title('Input Smoke')
        axes[0].axis('off')
        
        # 注意力权重矩阵
        im1 = axes[1].imshow(attn, cmap='viridis')
        axes[1].set_title('Attention Matrix')
        axes[1].set_xlabel('Key Position')
        axes[1].set_ylabel('Query Position')
        plt.colorbar(im1, ax=axes[1])
        
        # 平均注意力权重
        avg_attn = attn.mean(axis=0)
        sqrt_len = int(np.sqrt(len(avg_attn)))
        
        if sqrt_len * sqrt_len == len(avg_attn):
            avg_attn_2d = avg_attn.reshape(sqrt_len, sqrt_len)
            im2 = axes[2].imshow(avg_attn_2d, cmap='plasma')
            axes[2].set_title('Average Attention')
            axes[2].axis('off')
            plt.colorbar(im2, ax=axes[2])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
