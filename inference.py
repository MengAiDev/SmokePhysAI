import torch
import argparse
import yaml
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# Import custom modules
from src.utils.data_loader import SyntheticSmokeDataset
from src.utils.visualization import SmokeVisualizer
from src.models.smokephys_net import SmokePhysNet
from src.physics.smoke_simulator import SmokeSimulator

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

def generate_test_sequence(simulator, sequence_length=20):
    """Generate test sequence"""
    simulator.ns_solver.setup_grid()
    
    # Add smoke sources
    positions = [(64, 64), (32, 32), (96, 96)]
    intensities = [1.5, 1.0, 0.8]
    simulator.add_incense_source(positions, intensities)
    
    # Generate sequence
    sequence = []
    for _ in tqdm(range(sequence_length), desc="Generating smoke sequence"):
        density = simulator.simulate_step()
        sequence.append(density.clone().cpu().numpy())
    
    return sequence

def run_inference(model, sequence, device):
    """Run model inference"""
    predictions = []
    physics_features = []
    
    with torch.no_grad():
        for i in tqdm(range(len(sequence) - 1), desc="Running inference"):
            # Prepare input data
            input_frame = torch.tensor(sequence[i], device=device).unsqueeze(0).unsqueeze(0).float()
            
            # Model prediction
            output = model(input_frame)
            
            # Save results
            reconstructed = output['reconstructed'].squeeze().cpu().numpy()
            predictions.append(reconstructed)
            
            # Save physics features
            phys_feat = output['physics_features'].squeeze().cpu().numpy()
            physics_features.append(phys_feat)
    
    return predictions, physics_features

def visualize_results(ground_truth, predictions, physics_features):
    """Visualize results"""
    visualizer = SmokeVisualizer(figsize=(15, 10))
    
    # Visualize smoke evolution
    combined_sequence = ground_truth[1:]  # Start from second frame as target
    visualizer.plot_smoke_evolution(combined_sequence, save_path="ground_truth.png")
    visualizer.plot_smoke_evolution(predictions, save_path="predictions.png")
    
    # Visualize physics features
    chaos_metrics = {
        'lyapunov_exponent': [feat[0] for feat in physics_features],
        'fractal_dimension': [feat[1] for feat in physics_features],
        'entropy': [feat[2] for feat in physics_features]
    }
    visualizer.plot_chaos_features(chaos_metrics, save_path="physics_features.png")
    
    # Compare specific frames
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    frame_indices = [0, len(predictions)//2, -1]
    
    for i, idx in enumerate(frame_indices):
        # Ground truth frame
        axes[0, i].imshow(ground_truth[idx+1], cmap='hot')
        axes[0, i].set_title(f'Ground Truth Frame {idx+1}')
        axes[0, i].axis('off')
        
        # Predicted frame
        axes[1, i].imshow(predictions[idx], cmap='hot')
        axes[1, i].set_title(f'Predicted Frame {idx+1}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig("comparison.png", dpi=150)
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='SmokePhysAI Inference Script')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(config, args.checkpoint, str(device))
    
    # Create smoke simulator
    simulator = SmokeSimulator(
        grid_size=tuple(config['simulation']['grid_size']),
        dt=config['simulation']['dt'],
        viscosity=config['simulation']['viscosity'],
        device=str(device)
    )
    
    # Generate test sequence
    sequence = generate_test_sequence(simulator, sequence_length=20)
    
    # Run inference
    predictions, physics_features = run_inference(model, sequence, device)
    
    # Visualize results
    visualize_results(sequence, predictions, physics_features)
    print("Visualization results have been saved to current directory")

if __name__ == "__main__":
    main()
