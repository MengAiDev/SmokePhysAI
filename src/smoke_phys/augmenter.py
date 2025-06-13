# Physics augmentation pipeline
# Contains data augmentation methods with physical constraints

import torch
import numpy as np
from smoke_phys.simulator import SmokeSimulator
import torch.nn.functional as F

def calculate_fractal_dim(density_field):
    """计算分形维度"""
    threshold = 0.5
    binary = (density_field > threshold).float()
    
    sizes = 2**np.arange(2, 8)
    counts = []
    
    for size in sizes:
        pool = torch.nn.functional.adaptive_max_pool2d(
            binary.unsqueeze(0).unsqueeze(0), (size, size)
        )
        counts.append(pool.sum().item())
        
    log_sizes = np.log(sizes)
    log_counts = np.log(counts)
    slope, _ = np.polyfit(log_sizes, log_counts, 1)
    return -slope

class PhysicsAugmenter:
    def __init__(self, base_dataset, resolution=256, device='cpu'):
        self.base_dataset = base_dataset
        self.device = device
        self.sim = SmokeSimulator(resolution=resolution, device=device)
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        img = img.to(self.device)
        
        # Generate smoke texture in batches
        if idx % 100 == 0:
            self._precompute_smoke()
        
        smoke_tex = self.smoke_cache[idx % 100]
        
        # 计算分形参数
        alpha = calculate_fractal_dim(smoke_tex)
        
        # 物理融合
        augmented_img = self.physical_blend(img, smoke_tex, alpha)
        
        # 遮挡处理
        updated_label = self.adjust_bbox(label, smoke_tex)
        
        return augmented_img.cpu(), updated_label
    
    def _precompute_smoke(self):
        """Precompute smoke textures in batches"""
        self.smoke_cache = []
        for _ in range(100):
            pos = (np.random.uniform(0.3, 0.7), np.random.uniform(0.3, 0.7))
            self.sim.add_heat_source(pos, np.random.uniform(0.4, 0.8))
            self.smoke_cache.append(self.sim.run_steps(np.random.randint(20, 80)))
    
    def physical_blend(self, img, smoke, alpha):
        """物理启发的混合"""
        # 调整烟雾纹理尺寸以匹配输入图像
        smoke = F.interpolate(smoke.unsqueeze(0).unsqueeze(0), 
                            size=img.shape[-2:], 
                            mode='bilinear').squeeze()
        # 烟雾散射模型
        transmittance = torch.exp(-alpha * smoke)
        scattered_light = 0.8 * alpha * smoke
        return img * transmittance + scattered_light
    
    def adjust_bbox(self, bbox, smoke):
        """Adjust bounding box based on smoke density or pass through MNIST labels"""
        # If bbox is just a label (int), return it unchanged
        if isinstance(bbox, int):
            return torch.tensor(bbox, dtype=torch.float32)
            
        if isinstance(bbox, torch.Tensor):
            bbox = bbox.cpu().numpy()
            
        # Ensure bbox is iterable
        if not isinstance(bbox, (list, tuple, np.ndarray)):
            raise TypeError(f"Expected iterable bbox or label, got {type(bbox)}")
            
        # Original bbox adjustment logic
        x_min, y_min, x_max, y_max = map(int, bbox)
        roi = smoke[y_min:y_max, x_min:x_max]
        occlusion = 1 - roi.mean().item()
        
        if occlusion > 0.7:
            return torch.tensor([0, 0, 0, 0], dtype=torch.float32)
            
        shrink = int(occlusion * 5)
        return torch.tensor([
            max(0, x_min + shrink),
            max(0, y_min + shrink),
            min(self.sim.res, x_max - shrink),
            min(self.sim.res, y_max - shrink)
        ], dtype=torch.float32)