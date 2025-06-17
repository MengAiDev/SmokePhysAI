import torch
import torch.nn as nn
from physics_regularizer import PhysicsRegularizer
from chaos_attention import ChaosAttention

class SmokePhysNet(nn.Module):
    """SmokePhysAI主网络"""
    
    def __init__(self,
                 input_dim: int = 128,
                 hidden_dim: int = 512,
                 num_layers: int = 6,
                 num_heads: int = 8,
                 output_channels: int = 64,
                 chaos_strength: float = 0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 输入编码器
        self.input_encoder = nn.Sequential(
            nn.Conv2d(1, 64, 7, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((input_dim, input_dim))
        )
        
        # 位置编码
        self.pos_embedding = nn.Parameter(torch.randn(1, input_dim * input_dim, hidden_dim))
        
        # 特征投影
        self.feature_proj = nn.Linear(128, hidden_dim)
        
        # 混沌感知Transformer层
        self.chaos_layers = nn.ModuleList([
            ChaosTransformerLayer(
                hidden_dim, 
                num_heads, 
                chaos_strength=chaos_strength
            ) for _ in range(num_layers)
        ])
        
        # 输出解码器
        self.output_decoder = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, output_channels),
        )
        
        # 重建头
        self.reconstruction_head = nn.Sequential(
            nn.ConvTranspose2d(output_channels, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Sigmoid()
        )
        
        # 物理预测头
        self.physics_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 3),  # 预测混沌特征: lyapunov, fractal_dim, entropy
        )
        
        # 物理正则化器
        self.physics_regularizer = PhysicsRegularizer()
        
    def forward(self, x: torch.Tensor, return_features: bool = False) -> dict:
        """
        Args:
            x: [B, C, H, W] 输入烟雾图像
            return_features: 是否返回中间特征
        """
        B, C, H, W = x.shape
        
        # 1. 编码输入
        encoded = self.input_encoder(x)  # [B, 128, input_dim, input_dim]
        
        # 2. 展平并投影
        flattened = encoded.flatten(2).transpose(1, 2)  # [B, L, 128]
        features = self.feature_proj(flattened)  # [B, L, hidden_dim]
        
        # 3. 添加位置编码
        features = features + self.pos_embedding
        
        # 4. 通过混沌感知Transformer
        for layer in self.chaos_layers:
            features = layer(features)
            
        # 5. 输出预测
        output_features = self.output_decoder(features)  # [B, L, output_channels]
        
        # 6. 重建图像
        output_reshaped = output_features.transpose(1, 2).view(
            B, -1, self.input_dim, self.input_dim
        )
        reconstructed = self.reconstruction_head(output_reshaped)
        
        # 7. 物理特征预测
        pooled_features = features.mean(dim=1)  # [B, hidden_dim]
        physics_pred = self.physics_head(pooled_features)
        
        results = {
            'reconstructed': reconstructed,
            'physics_features': physics_pred,
            'latent_features': pooled_features
        }
        
        if return_features:
            results['intermediate_features'] = features
            
        return results


class ChaosTransformerLayer(nn.Module):
    """混沌感知Transformer层"""
    
    def __init__(self, 
                 dim: int, 
                 num_heads: int,
                 chaos_strength: float = 0.1,
                 dropout: float = 0.1):
        super().__init__()
        
        self.chaos_attention = ChaosAttention(
            dim, num_heads, chaos_strength
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 混沌注意力 + 残差连接
        x = x + self.chaos_attention(self.norm1(x))
        
        # FFN + 残差连接
        x = x + self.ffn(self.norm2(x))
        
        return x