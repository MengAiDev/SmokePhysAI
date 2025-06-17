import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from typing import List, Dict, Callable

class PerturbationTester:
    """扰动测试器"""
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        
    def gaussian_noise_test(self,
                           model: nn.Module,
                           test_data: torch.Tensor,
                           noise_levels: List[float] = [0.01, 0.05, 0.1, 0.2]) -> Dict:
        """高斯噪声鲁棒性测试"""
        model.eval()
        results = {}
        
        # 基线性能
        with torch.no_grad():
            baseline = model(test_data)
            baseline_features = baseline['latent_features']
            
        for noise_level in noise_levels:
            # 添加高斯噪声
            noise = torch.randn_like(test_data) * noise_level
            noisy_data = test_data + noise
            noisy_data = torch.clamp(noisy_data, 0, 1)
            
            with torch.no_grad():
                noisy_pred = model(noisy_data)
                noisy_features = noisy_pred['latent_features']
                
            # 计算特征稳定性
            feature_stability = F.cosine_similarity(
                baseline_features, noisy_features, dim=1
            ).mean().item()
            
            results[f'gaussian_{noise_level}'] = {
                'feature_stability': feature_stability,
                'reconstruction_mse': F.mse_loss(
                    noisy_pred['reconstructed'], 
                    baseline['reconstructed']
                ).item()
            }
            
        return results
        
    def adversarial_test(self,
                        model: nn.Module,
                        test_data: torch.Tensor,
                        epsilon: float = 0.1,
                        num_steps: int = 10) -> Dict:
        """对抗样本鲁棒性测试"""
        model.eval()
        
        # PGD攻击
        delta = torch.zeros_like(test_data, requires_grad=True)
        
        for _ in range(num_steps):
            with torch.enable_grad():
                adversarial_data = test_data + delta
                adversarial_data = torch.clamp(adversarial_data, 0, 1)
                
                output = model(adversarial_data)
                
                # 最大化重建误差
                loss = -F.mse_loss(output['reconstructed'], test_data)
                loss.backward()
                
            # 更新扰动
            delta_grad = delta.grad.data # type: ignore
            delta.data = delta.data + epsilon/num_steps * torch.sign(delta_grad)
            delta.data = torch.clamp(delta.data, -epsilon, epsilon)
            delta.grad.zero_() # type: ignore
            
        # 评估对抗样本
        with torch.no_grad():
            baseline = model(test_data)
            adversarial_output = model(torch.clamp(test_data + delta, 0, 1))
            
            feature_stability = F.cosine_similarity(
                baseline['latent_features'],
                adversarial_output['latent_features'],
                dim=1
            ).mean().item()
            
        return {
            'adversarial_feature_stability': feature_stability,
            'adversarial_perturbation_norm': torch.norm(delta).item()
        }
        
    def physics_perturbation_test(self,
                                 model: nn.Module,
                                 simulator,
                                 num_tests: int = 50) -> Dict:
        """物理扰动测试"""
        model.eval()
        results = []
        
        for _ in range(num_tests):
            # 生成随机烟雾场景
            simulator.ns_solver.setup_grid()
            
            # 添加随机源
            num_sources = np.random.randint(1, 4)
            for _ in range(num_sources):
                x = np.random.randint(20, simulator.ns_solver.w - 20)
                y = np.random.randint(20, simulator.ns_solver.h - 20)
                intensity = np.random.uniform(0.5, 2.0)
                simulator.add_incense_source([(x, y)], [intensity])
                
            # 仿真序列
            sequence = []
            for t in range(20):
                density = simulator.simulate_step()
                sequence.append(density.unsqueeze(0).unsqueeze(0))
                
            # 评估模型预测一致性
            with torch.no_grad():
                predictions = []
                for frame in sequence:
                    pred = model(frame)
                    predictions.append(pred['physics_features'])
                    
            # 计算预测稳定性
            pred_tensor = torch.stack(predictions)
            pred_var = torch.var(pred_tensor, dim=0).mean().item()
            
            results.append({
                'prediction_variance': pred_var,
                'sequence_length': len(sequence)
            })
            
        avg_variance = np.mean([r['prediction_variance'] for r in results])
        
        return {
            'physics_prediction_stability': 1.0 / (1.0 + avg_variance),
            'num_tests': num_tests
        }
