import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ChaosAttention(nn.Module):
    """基于混沌动力学的注意力机制"""
    
    def __init__(self, 
                 dim: int,
                 num_heads: int = 8,
                 chaos_strength: float = 0.1,
                 temperature: float = 1.0):
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.chaos_strength = chaos_strength
        self.temperature = temperature
        
        assert dim % num_heads == 0
        
        # 标准注意力组件
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        # 混沌动力学组件
        self.chaos_proj = nn.Linear(dim, dim)
        self.chaos_gate = nn.Linear(dim, 1)
        
        # Lorenz系统参数
        self.register_buffer('lorenz_sigma', torch.tensor(10.0))
        self.register_buffer('lorenz_rho', torch.tensor(28.0))
        self.register_buffer('lorenz_beta', torch.tensor(8.0/3.0))
        
    def lorenz_system(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, dt: float = 0.01) -> tuple:
        """Lorenz混沌系统"""
        dx = self.lorenz_sigma * (y - x)
        dy = x * (self.lorenz_rho - z) - y
        dz = x * y - self.lorenz_beta * z
        
        return (x + dt * dx, y + dt * dy, z + dt * dz)
        
    def generate_chaos_field(self, seq_len: int, batch_size: int, device: str) -> torch.Tensor:
        """生成混沌场"""
        # 初始化Lorenz系统
        x = torch.randn(batch_size, seq_len, device=device)
        y = torch.randn(batch_size, seq_len, device=device)
        z = torch.randn(batch_size, seq_len, device=device)
        
        # 迭代生成混沌轨迹
        chaos_seq = []
        for _ in range(10):  # 迭代10次
            x, y, z = self.lorenz_system(x, y, z)
            chaos_seq.append(torch.stack([x, y, z], dim=-1))
            
        # 组合混沌特征
        chaos_field = torch.stack(chaos_seq, dim=1).mean(dim=1)  # [B, L, 3]
        
        return chaos_field
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: [B, L, D] 输入序列
            mask: [B, L] 注意力掩码
        """
        B, L, D = x.shape
        
        # 标准注意力计算
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 生成混沌场
        chaos_field = self.generate_chaos_field(L, B, x.device)
        chaos_features = self.chaos_proj(chaos_field)  # [B, L, D]
        
        # 混沌门控
        chaos_gate = torch.sigmoid(self.chaos_gate(chaos_features))  # [B, L, 1]
        
        # 将混沌特征注入注意力
        chaos_scores = torch.matmul(
            chaos_features.view(B, L, self.num_heads, self.head_dim).transpose(1, 2),
            k.transpose(-2, -1)
        ) / math.sqrt(self.head_dim)
        
        # 组合传统注意力和混沌注意力
        final_scores = scores + self.chaos_strength * chaos_scores * chaos_gate.transpose(-2, -1)
        
        # 应用掩码
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, L]
            final_scores = final_scores.masked_fill(mask == 0, float('-inf'))
            
        # Softmax
        attn_weights = F.softmax(final_scores / self.temperature, dim=-1)
        
        # 应用注意力
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        
        return self.out_proj(out)