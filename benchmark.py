"""
Benchmark comparison between SmokePhysAI and traditional computer vision models
for smoke sequence prediction and analysis.
"""

import torch
import argparse
import yaml
import time
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

# Import custom modules
from src.utils.data_loader import SyntheticSmokeDataset
from src.models.smokephys_net import SmokePhysNet

# Traditional CV methods
def farneback_optical_flow(prev_frame, next_frame):
    """Compute optical flow using Farneback method"""
    # Handle grayscale images (1 channel)
    if prev_frame.ndim == 3 and prev_frame.shape[2] == 1:
        prev_gray = prev_frame[:,:,0]
        next_gray = next_frame[:,:,0]
    # Handle color images (3 channels)
    elif prev_frame.ndim == 3 and prev_frame.shape[2] == 3:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
    # Already grayscale (2D)
    else:
        prev_gray = prev_frame
        next_gray = next_frame
        
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0 #type: ignore
    ) # type: ignore
    return flow

def lucas_kanade_optical_flow(prev_frame, next_frame):
    """Compute optical flow using Lucas-Kanade method"""
    # Handle grayscale images (1 channel)
    if prev_frame.ndim == 3 and prev_frame.shape[2] == 1:
        prev_gray = prev_frame[:,:,0]
        next_gray = next_frame[:,:,0]
    # Handle color images (3 channels)
    elif prev_frame.ndim == 3 and prev_frame.shape[2] == 3:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
    # Already grayscale (2D)
    else:
        prev_gray = prev_frame
        next_gray = next_frame
    
    # Feature detection
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params) # type: ignore
    
    if p0 is None:
        return np.zeros((*prev_gray.shape, 2), dtype=np.float32)
    
    # Calculate optical flow
    p1, st, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, next_gray, p0, None, winSize=(15, 15), maxLevel=2 # type: ignore
    ) # type: ignore
    
    # Create flow field
    flow = np.zeros((*prev_gray.shape, 2), dtype=np.float32)
    if p1 is not None:
        good_new = p1[st == 1]
        good_old = p0[st == 1]
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            x0, y0 = old.ravel()
            x1, y1 = new.ravel()
            flow[int(y0), int(x0)] = [x1 - x0, y1 - y0]
    
    return flow

def predict_next_frame(prev_frame, flow):
    """Predict next frame using optical flow"""
    h, w = prev_frame.shape[:2]
    map_x = np.tile(np.arange(w), (h, 1)).astype(np.float32)
    map_y = np.swapaxes(np.tile(np.arange(h), (w, 1)), 0, 1).astype(np.float32)
    
    # Apply flow vectors
    map_x += flow[..., 0]
    map_y += flow[..., 1]
    
    # Warp image
    predicted = cv2.remap(
        prev_frame, map_x, map_y, interpolation=cv2.INTER_LINEAR
    )
    return predicted

def load_config(config_path: str) -> dict:
    """Load configuration file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_model(config: dict, checkpoint_path: str, device: str) -> SmokePhysNet:
    """Load pre-trained model"""
    model = SmokePhysNet(
        input_dim=config['model']['input_dim'],
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers'],
        num_heads=config['model']['num_heads'],
        chaos_strength=config['model']['chaos_strength']
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def evaluate_model(model, test_loader, device):
    """Evaluate SmokePhysAI model performance"""
    model.eval()
    total_mse = 0.0
    total_ssim = 0.0
    total_time = 0.0
    physics_corr = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating SmokePhysAI"):
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            chaos_targets = batch['chaos_features'].to(device)
            
            # Time prediction
            start_time = time.time()
            outputs = model(inputs)
            total_time += time.time() - start_time
            
            # Calculate reconstruction metrics
            recon = outputs['reconstructed']
            mse = torch.nn.functional.mse_loss(recon, targets).item()
            total_mse += mse
            
            # Calculate physics feature correlation
            phys_pred = outputs['physics_features']
            for i in range(phys_pred.shape[0]):
                corr = pearsonr(
                    phys_pred[i].cpu().numpy(), 
                    chaos_targets[i].cpu().numpy()
                )[0]
                physics_corr.append(corr)
    
    avg_mse = total_mse / len(test_loader)
    avg_ssim = total_ssim / len(test_loader)
    avg_phys_corr = np.mean(physics_corr)
    avg_time = total_time / len(test_loader.dataset)
    
    return {
        'mse': avg_mse,
        'ssim': avg_ssim,
        'physics_correlation': avg_phys_corr,
        'inference_time': avg_time
    }

def evaluate_traditional_cv(test_loader):
    """Evaluate traditional computer vision methods"""
    farneback_results = {'mse': [], 'time': []}
    lucas_kanade_results = {'mse': [], 'time': []}
    
    for batch in tqdm(test_loader, desc="Evaluating Traditional CV"):
        # Convert tensors to numpy arrays
        prev_frames = batch['input'].permute(0, 2, 3, 1).cpu().numpy() * 255
        next_frames = batch['target'].permute(0, 2, 3, 1).cpu().numpy() * 255
        
        for i in range(prev_frames.shape[0]):
            prev_frame = prev_frames[i].astype(np.uint8)
            next_frame = next_frames[i].astype(np.uint8)
            
            # Farneback method
            start_time = time.time()
            flow_farneback = farneback_optical_flow(prev_frame, next_frame)
            pred_farneback = predict_next_frame(prev_frame, flow_farneback)
            farneback_time = time.time() - start_time
            
            # Lucas-Kanade method
            start_time = time.time()
            flow_lk = lucas_kanade_optical_flow(prev_frame, next_frame)
            pred_lk = predict_next_frame(prev_frame, flow_lk)
            lk_time = time.time() - start_time
            
            # Calculate MSE
            mse_farneback = mean_squared_error(
                next_frame.flatten(), 
                pred_farneback.flatten()
            )
            mse_lk = mean_squared_error(
                next_frame.flatten(), 
                pred_lk.flatten()
            )
            
            farneback_results['mse'].append(mse_farneback)
            farneback_results['time'].append(farneback_time)
            lucas_kanade_results['mse'].append(mse_lk)
            lucas_kanade_results['time'].append(lk_time)
    
    return {
        'Farneback': {
            'mse': np.mean(farneback_results['mse']),
            'inference_time': np.mean(farneback_results['time'])
        },
        'Lucas-Kanade': {
            'mse': np.mean(lucas_kanade_results['mse']),
            'inference_time': np.mean(lucas_kanade_results['time'])
        }
    }

def print_results(model_results, cv_results):
    """Print benchmark results in a table format"""
    print("\n" + "="*60)
    print(f"{'Model':<20} | {'MSE':<15} | {'Physics Corr':<15} | {'Inference Time (ms)':<15}")
    print("-"*60)
    
    # Print SmokePhysAI results
    print(f"{'SmokePhysAI':<20} | "
          f"{model_results['mse']:.6f} | "
          f"{model_results['physics_correlation']:.4f} | "
          f"{model_results['inference_time']*1000:.2f}")
    
    # Print traditional CV results
    for method, results in cv_results.items():
        print(f"{method:<20} | "
              f"{results['mse']:.6f} | "
              f"{'N/A':<15} | "
              f"{results['inference_time']*1000:.2f}")
    
    print("="*60)
    print("Note: Physics Correlation measures how well the model predicts chaos features")
    print("      (Lyapunov exponent, Fractal dimension, Entropy) compared to ground truth")

def main():
    parser = argparse.ArgumentParser(description='SmokePhysAI Benchmark')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--num_samples', type=int, default=50,
                       help='Number of test samples to evaluate')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(config, args.checkpoint, str(device))
    
    # Create test dataset
    test_dataset = SyntheticSmokeDataset(
        num_samples=args.num_samples,
        grid_size=tuple(config['data']['grid_size']),
        device='cpu'  # Keep on CPU for traditional CV methods
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=4, shuffle=False
    )
    
    # Evaluate SmokePhysAI
    print("\nEvaluating SmokePhysAI model...")
    model_results = evaluate_model(model, test_loader, device)
    
    # Evaluate traditional CV methods
    print("\nEvaluating traditional computer vision methods...")
    cv_results = evaluate_traditional_cv(test_loader)
    
    # Print results
    print_results(model_results, cv_results)

if __name__ == "__main__":
    main()
