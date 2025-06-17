# SmokePhysAI

An innovative computer vision project that leverages the physical properties of aromatic smoke for advanced physical simulations and visual analysis.

## Overview
SmokePhysAI combines computer vision techniques with fluid dynamics to analyze and simulate the behavior of aromatic smoke. This project enables:

- Real-time smoke pattern recognition
- Physical property extraction from smoke movements
- Predictive modeling of smoke dispersion
- Visual enhancement of smoke dynamics

## Features
- 🌀 **Smoke Tracking**: Advanced particle tracking algorithms
- 🔍 **Property Analysis**: Density, velocity, and turbulence measurement
- 🤖 **AI Simulation**: Predictive modeling using physics-informed neural networks
- 📊 **Data Visualization**: Interactive 3D smoke visualization tools

## Performance
Benchmark results comparing SmokePhysAI with traditional computer vision methods:

| Model                | MSE             | Physics Correlation | Inference Time (ms) |
|----------------------|-----------------|---------------------|---------------------|
| **SmokePhysAI**      | 0.002955        | 0.9957              | 610.92              |
| Farneback            | 0.699607        | N/A                 | 3.98                |
| Lucas-Kanade         | 0.723172        | N/A                 | 0.71                |

**Notes:**
- **MSE** (Mean Squared Error) measures reconstruction accuracy (lower is better)
- **Physics Correlation** measures accuracy of physical property prediction (1.0 is perfect)
- **Inference Time** is per-frame processing time (lower is better)

SmokePhysAI achieves **200x higher accuracy** than traditional methods while capturing physical properties with near-perfect correlation. We're actively working to optimize inference speed.

## Installation
```bash
git clone https://github.com/MengAiDev/SmokePhysAI.git
cd SmokePhysAI
pip install -e .
```

## Pre-trained Models
You can download pre-trained models from ModelScope:
- 🤖 [SmokePhysAI-Pretrained](https://modelscope.cn/models/MengAiDev/SmokePhysAI-Pretrained/)

## Usage
Train from scratch:
```bash
python train.py --config config/config.yaml
```

Run inference:
```bash
python inference.py --model_path model.pth --config config/config.yaml
```

Run benchmark:
```bash
python benchmark.py --checkpoint model.pth --num_samples 100
```

Please see the inference_output/ directory for the results, which trained by myself on RTX 3090.

## Contributing
Contributions are welcome!

## License
This project is licensed under the Apache 2.0 License - see [LICENSE](LICENSE) for details.

