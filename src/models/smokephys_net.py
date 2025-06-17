import torch
import torch.nn as nn
import torch.nn.functional as F
from .physics_regularizer import PhysicsRegularizer
from .chaos_attention import ChaosAttention

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
        encoded = self.input_encoder(x)  # [B, 128, self.input_dim, self.input_dim]
        
        # 新增：降低分辨率以减少内存使用
        reduced_size = 32  # 将空间维度减少到32x32
        encoded = F.adaptive_avg_pool2d(encoded, (reduced_size, reduced_size))
        pool_size = reduced_size
        
        # 2. 展平并投影
        flattened = encoded.flatten(2).transpose(1, 2)  # [B, pool_size*pool_size, 128]
        features = self.feature_proj(flattened)  # [B, pool_size*pool_size, hidden_dim]
        
        # 3. 添加位置编码：若token数发生变化，则对位置编码进行插值
        expected_tokens = pool_size * pool_size
        if expected_tokens != self.pos_embedding.shape[1]:
            pos_embed = self.pos_embedding.reshape(1, self.input_dim, self.input_dim, self.hidden_dim).permute(0, 3, 1, 2)
            pos_embed = torch.nn.functional.interpolate(pos_embed, size=(pool_size, pool_size), mode='bilinear', align_corners=False)
            pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(1, expected_tokens, self.hidden_dim)
        else:
            pos_embed = self.pos_embedding
        
        features = features + pos_embed
        
        # 4. 通过混沌感知Transformer
        for layer in self.chaos_layers:
            features = layer(features)
            
        # 5. 输出预测
        output_features = self.output_decoder(features)
        # 6. 重建图像
        output_reshaped = output_features.transpose(1, 2).view(B, -1, pool_size, pool_size)
        reconstructed = self.reconstruction_head(output_reshaped)
        
        # 7. 物理特征预测
        pooled_features = features.mean(dim=1)
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