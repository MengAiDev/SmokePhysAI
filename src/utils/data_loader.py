import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os  # 新增：用于根据 CPU 核心数动态调整
from typing import Tuple, List, Optional, Dict
from tqdm import tqdm  # 新增导入
import pickle  # 新增


class SyntheticSmokeDataset(Dataset):
    """合成烟雾数据集"""
    
    def __init__(self,
                 num_samples: int = 1000,
                 grid_size: Tuple[int, int] = (128, 128),
                 sequence_length: int = 20,
                 device: str = 'cuda',
                 cache_path: Optional[str] = None):  # 新增参数
        self.num_samples = num_samples
        self.grid_size = grid_size
        self.sequence_length = sequence_length
        self.device = device
        self.cache_path = cache_path  # 保存缓存路径
        
        if self.cache_path and os.path.exists(self.cache_path):
            with open(self.cache_path, 'rb') as f:
                self.data = pickle.load(f)
            print(f"Loaded synthetic data from {self.cache_path}")
        else:
            self.data = self._generate_synthetic_data()
            if self.cache_path:
                os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)  # 新增：确保目录存在
                with open(self.cache_path, 'wb') as f:
                    pickle.dump(self.data, f)
                print(f"Saved synthetic data to {self.cache_path}")
                
    def _generate_synthetic_data(self) -> List[Dict]:
        """生成合成数据"""
        from ..physics.smoke_simulator import SmokeSimulator
        
        data = []
        simulator = SmokeSimulator(self.grid_size, device=self.device)
        
        for i in tqdm(range(self.num_samples), desc="Generating synthetic smoke samples"):  # 修改处
            # 重置仿真器
            simulator.ns_solver.setup_grid()
            
            # 随机源配置
            num_sources = np.random.randint(1, 4)
            positions = []
            intensities = []
            
            for _ in range(num_sources):
                x = np.random.randint(20, self.grid_size[1] - 20)
                y = np.random.randint(20, self.grid_size[0] - 20)
                intensity = np.random.uniform(0.5, 2.0)
                positions.append((x, y))
                intensities.append(intensity)
                
            simulator.add_incense_source(positions, intensities)
            
            # 生成序列
            sequence = []
            chaos_features = []
            
            for t in range(self.sequence_length):
                density = simulator.simulate_step()
                sequence.append(density.clone())
                
                # 获取混沌特征
                if t >= 10:  # 等待稳定
                    features = simulator.get_chaos_features()
                    if features:
                        chaos_features.append(features)
                        
            # 计算平均混沌特征
            if chaos_features:
                avg_chaos = {
                    'lyapunov_exponent': np.mean([f.get('lyapunov_exponent', 0) for f in chaos_features]),
                    'fractal_dimension': np.mean([f.get('fractal_dimension', 1) for f in chaos_features]),
                    'entropy': np.mean([f.get('entropy', 0) for f in chaos_features])
                }
            else:
                avg_chaos = {
                    'lyapunov_exponent': 0.0,
                    'fractal_dimension': 1.0,
                    'entropy': 0.0
                }
                
            data.append({
                'sequence': torch.stack(sequence),
                'chaos_features': avg_chaos,
                'source_config': {
                    'positions': positions,
                    'intensities': intensities
                }
            })
            # ...原有进度打印代码已移除...
                
        return data
        
    def __len__(self) -> int:
        return len(self.data)
        
    def __getitem__(self, idx: int) -> Dict:
        sample = self.data[idx]
        
        # 随机选择一帧作为输入
        frame_idx = np.random.randint(5, self.sequence_length - 5)
        input_frame = sample['sequence'][frame_idx].unsqueeze(0)  # 添加通道维度
        
        # 目标是下一帧
        target_frame = sample['sequence'][frame_idx + 1].unsqueeze(0)
        
        return {
            'input': input_frame,
            'target': target_frame,
            'chaos_features': torch.tensor([
                sample['chaos_features']['lyapunov_exponent'],
                sample['chaos_features']['fractal_dimension'],
                sample['chaos_features']['entropy']
            ], dtype=torch.float32),
            'sequence': sample['sequence']
        }


def create_data_loaders(batch_size: int = 16,
                        num_train: int = 800,
                        num_val: int = 200,
                        grid_size: Tuple[int, int] = (128, 128),
                        device: str = 'cuda',
                        cache_dir: Optional[str] = None) -> Tuple[DataLoader, DataLoader]:
    """创建训练和验证数据加载器"""
    
    # 新增：根据 CPU 核心数设置 num_workers
    cpu_count = os.cpu_count()
    num_workers = cpu_count if cpu_count is not None else 0

    # 根据设备调整 num_workers 和 pin_memory
    if (isinstance(device, str) and device.lower() == "cpu") or (isinstance(device, torch.device) and device.type == "cpu"):
        num_workers = 0
        pin_memory_val = False
    else:
        pin_memory_val = True

    if cache_dir:
        train_cache = os.path.join(cache_dir, "train_data.pkl")
        val_cache = os.path.join(cache_dir, "val_data.pkl")
    else:
        train_cache = None
        val_cache = None
    
    # 训练集
    train_dataset = SyntheticSmokeDataset(
        num_samples=num_train,
        grid_size=grid_size,
        device=device,
        cache_path=train_cache  # 传递缓存路径
    )
    
    # 验证集
    val_dataset = SyntheticSmokeDataset(
        num_samples=num_val,
        grid_size=grid_size,
        device=device,
        cache_path=val_cache  # 传递缓存路径
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,  # 修改：根据设备设定的 num_workers
        pin_memory=pin_memory_val  # 修改：根据设备设置是否使用 pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,  # 修改：根据设备设定的 num_workers
        pin_memory=pin_memory_val  # 修改：根据设备设置是否使用 pin_memory
    )
    
    return train_loader, val_loader

