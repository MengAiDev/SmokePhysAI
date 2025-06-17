import torch
import torch.nn as nn
import torch.nn.functional as F

class PhysicsRegularizer(nn.Module):
    """物理正则化约束"""
    
    def __init__(self, 
                 conservation_weight: float = 1.0,
                 continuity_weight: float = 1.0,
                 energy_weight: float = 0.5):
        super().__init__()
        
        self.conservation_weight = conservation_weight
        self.continuity_weight = continuity_weight
        self.energy_weight = energy_weight
        
    def mass_conservation_loss(self, density_pred: torch.Tensor, density_target: torch.Tensor) -> torch.Tensor:
        """质量守恒约束"""
        # 计算质量差异
        mass_pred = density_pred.sum(dim=(-2, -1))
        mass_target = density_target.sum(dim=(-2, -1))
        
        return F.mse_loss(mass_pred, mass_target)
        
    def continuity_loss(self, density_sequence: torch.Tensor) -> torch.Tensor:
        """连续性约束 - 相邻帧应该平滑变化"""
        if density_sequence.shape[1] < 2:
            return torch.tensor(0.0, device=density_sequence.device)
            
        # 计算时间梯度
        time_grad = density_sequence[:, 1:] - density_sequence[:, :-1]
        
        # 平滑性损失
        return torch.mean(torch.abs(time_grad))
        
    def energy_conservation_loss(self, velocity_pred: torch.Tensor) -> torch.Tensor:
        """能量守恒约束"""
        # 计算动能
        kinetic_energy = 0.5 * (velocity_pred**2).sum(dim=1)
        
        # 能量应该逐渐衰减
        if kinetic_energy.shape[0] > 1:
            energy_diff = kinetic_energy[1:] - kinetic_energy[:-1]
            # 惩罚能量增加
            energy_increase = torch.relu(energy_diff)
            return energy_increase.mean()
        
        return torch.tensor(0.0, device=velocity_pred.device)
        
    def divergence_loss(self, velocity: torch.Tensor) -> torch.Tensor:
        """散度约束 - 不可压缩流体的散度应为零"""
        if velocity.shape[1] != 2:  # 需要2D速度场
            return torch.tensor(0.0, device=velocity.device)
            
        u, v = velocity[:, 0], velocity[:, 1]
        
        # 计算散度
        du_dx = u[:, :, 1:] - u[:, :, :-1]
        dv_dy = v[:, 1:, :] - v[:, :-1, :]
        
        # 确保尺寸匹配
        min_h = min(du_dx.shape[1], dv_dy.shape[1])
        min_w = min(du_dx.shape[2], dv_dy.shape[2])
        
        du_dx = du_dx[:, :min_h, :min_w]
        dv_dy = dv_dy[:, :min_h, :min_w]
        
        divergence = du_dx + dv_dy
        
        return torch.mean(divergence**2)
        
    def forward(self, 
                predictions: dict,
                targets: dict = None) -> dict: # type: ignore
        """
        Args:
            predictions: 包含预测结果的字典
            targets: 包含目标值的字典
        """
        losses = {}
        total_loss = 0.0
        
        # 质量守恒
        if 'density' in predictions and targets and 'density' in targets:
            mass_loss = self.mass_conservation_loss(predictions['density'], targets['density'])
            losses['mass_conservation'] = mass_loss
            total_loss += self.conservation_weight * mass_loss
            
        # 连续性约束
        if 'density_sequence' in predictions:
            continuity_loss = self.continuity_loss(predictions['density_sequence'])
            losses['continuity'] = continuity_loss
            total_loss += self.continuity_weight * continuity_loss
            
        # 能量守恒
        if 'velocity' in predictions:
            energy_loss = self.energy_conservation_loss(predictions['velocity'])
            losses['energy_conservation'] = energy_loss
            total_loss += self.energy_weight * energy_loss
            
        # 散度约束
        if 'velocity' in predictions:
            div_loss = self.divergence_loss(predictions['velocity'])
            losses['divergence'] = div_loss
            total_loss += 0.5 * div_loss
            
        losses['total_physics_loss'] = total_loss
        
        return losses