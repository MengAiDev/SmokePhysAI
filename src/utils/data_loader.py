import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os  # Added: For dynamically adjusting based on CPU cores
from typing import Tuple, List, Optional, Dict
from tqdm import tqdm  # Added import
import pickle  # Added


class SyntheticSmokeDataset(Dataset):
    """Synthetic smoke dataset"""
    
    def __init__(self,
                 num_samples: int = 1000,
                 grid_size: Tuple[int, int] = (128, 128),
                 sequence_length: int = 20,
                 device: str = 'cuda',
                 cache_path: Optional[str] = None):  # Added parameter
        self.num_samples = num_samples
        self.grid_size = grid_size
        self.sequence_length = sequence_length
        self.device = device
        self.cache_path = cache_path  # Save cache path
        
        if self.cache_path and os.path.exists(self.cache_path):
            with open(self.cache_path, 'rb') as f:
                self.data = pickle.load(f)
            print(f"Loaded synthetic data from {self.cache_path}")
        else:
            self.data = self._generate_synthetic_data()
            if self.cache_path:
                os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)  # Added: Ensure directory exists
                with open(self.cache_path, 'wb') as f:
                    pickle.dump(self.data, f)
                print(f"Saved synthetic data to {self.cache_path}")
                
    def _generate_synthetic_data(self) -> List[Dict]:
        """Generate synthetic data"""
        from ..physics.smoke_simulator import SmokeSimulator
        
        data = []
        simulator = SmokeSimulator(self.grid_size, device=self.device)
        
        for i in tqdm(range(self.num_samples), desc="Generating synthetic smoke samples"):  # Modified
            # Reset simulator
            simulator.ns_solver.setup_grid()
            
            # Random source configuration
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
            
            # Generate sequence
            sequence = []
            chaos_features = []
            
            for t in range(self.sequence_length):
                density = simulator.simulate_step()
                sequence.append(density.clone())
                
                # Get chaos features
                if t >= 10:  # Wait for stabilization
                    features = simulator.get_chaos_features()
                    if features:
                        chaos_features.append(features)
                        
            # Calculate average chaos features
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
                
        return data
        
    def __len__(self) -> int:
        return len(self.data)
        
    def __getitem__(self, idx: int) -> Dict:
        sample = self.data[idx]
        
        # Randomly select a frame as input
        frame_idx = np.random.randint(5, self.sequence_length - 5)
        input_frame = sample['sequence'][frame_idx].unsqueeze(0)  # Add channel dimension
        
        # Target is next frame
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
    """Create training and validation data loaders"""
    
    # Added: Set num_workers based on CPU cores
    cpu_count = os.cpu_count()
    num_workers = cpu_count if cpu_count is not None else 0

    # Adjust num_workers and pin_memory based on device
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
    
    # Training set
    train_dataset = SyntheticSmokeDataset(
        num_samples=num_train,
        grid_size=grid_size,
        device=device,
        cache_path=train_cache  # Pass cache path
    )
    
    # Validation set
    val_dataset = SyntheticSmokeDataset(
        num_samples=num_val,
        grid_size=grid_size,
        device=device,
        cache_path=val_cache  # Pass cache path
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,  # Modified: Use device-based num_workers
        pin_memory=pin_memory_val  # Modified: Use device-based pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,  # Modified: Use device-based num_workers
        pin_memory=pin_memory_val  # Modified: Use device-based pin_memory
    )
    
    return train_loader, val_loader

