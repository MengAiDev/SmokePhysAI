# Chaotic attention mechanism
# Implements physics-aware attention for smoke modeling

import torch
import torch.nn as nn
import torch.nn.functional as F

class PhysicsGuidedAttention(nn.Module):
    def __init__(self, in_channels, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        self.query_conv = nn.Conv2d(in_channels, in_channels//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        # 分形正则化参数
        self.fractal_weight = nn.Parameter(torch.tensor(0.5))
        
    def compute_fractal_regularizer(self, attention_map):
        """基于分形几何的正则化"""
        # 确保attention_map是4D张量 [batch, channels, height, width]
        if attention_map.dim() == 3:
            attention_map = attention_map.unsqueeze(1)
            
        batch_size, _, h, w = attention_map.shape
        
        # 计算多尺度方差
        scales = [1, 2, 4, 8]
        variances = []
        
        for scale in scales:
            if h % scale == 0 and w % scale == 0:
                pool = F.avg_pool2d(attention_map, scale, scale)
                variances.append(pool.var(dim=(2,3)))
                
        if len(variances) < 2:
            return torch.tensor(0.0, device=attention_map.device)
            
        variance_ratio = torch.stack(variances[:-1]) / torch.stack(variances[1:])
        return variance_ratio.mean()
        
    def forward(self, x):
        batch_size, C, H, W = x.size()
        
        # 标准注意力计算
        query = self.query_conv(x).view(batch_size, -1, H*W).permute(0,2,1)
        key = self.key_conv(x).view(batch_size, -1, H*W)
        value = self.value_conv(x).view(batch_size, -1, H*W)
        
        attention = torch.bmm(query, key) * self.temperature
        attention = F.softmax(attention, dim=-1)
        
        # 添加混沌扰动
        chaos = torch.randn_like(attention) * 0.1 * self.fractal_weight
        attention = F.softmax(attention + chaos, dim=-1)
        
        out = torch.bmm(value, attention.permute(0,2,1))
        out = out.view(batch_size, C, H, W)
        
        # 应用分形正则化
        # 修正形状计算错误，根据实际attention维度调整
        attention_4d = attention.view(batch_size, 1, H*W, H*W).mean(dim=-1).view(batch_size, 1, H, W)
        reg = self.compute_fractal_regularizer(attention_4d)
        return out, reg