# Inference interface
# Provides API for model prediction and visualization

import torch
import cv2
import numpy as np
from torchvision import transforms
import torch.nn.functional as F
from smoke_phys.simulator import SmokeSimulator
from smoke_phys.augmenter import calculate_fractal_dim
from models.model import PhysicsAwareNet

class PhysicsInference:
    def __init__(self, model_path, resolution=128, device=None):
        # 允许用户指定设备，默认自动选择
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_path)
        self.sim = SmokeSimulator(resolution=resolution, device=self.device)
        
    def _load_model(self, path):
        model = PhysicsAwareNet().to(self.device)
        # 支持从CPU保存的模型加载
        state_dict = torch.load(path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.eval()
        return model
    
    def preprocess(self, image):
        # 图像预处理
        img_tensor = transforms.ToTensor()(image)
        img_tensor = transforms.Normalize((0.5,), (0.5,))(img_tensor)
        return img_tensor.unsqueeze(0).to(self.device)
    
    def apply_physical_perturbation(self, image, steps=50):
        # 添加物理扰动
        self.sim.add_heat_source((0.5, 0.5), 0.8)
        smoke_tex = self.sim.run_steps(steps)
        
        # 计算分形参数
        alpha = calculate_fractal_dim(smoke_tex)
        
        # 应用扰动
        perturbed = image * torch.exp(-alpha * smoke_tex.expand_as(image))
        return perturbed
    
    def predict(self, image):
        # 预处理
        input_tensor = self.preprocess(image)
        
        # 推理
        with torch.no_grad():
            output, _ = self.model(input_tensor)
            prob = F.softmax(output, dim=1)
            
        return prob.argmax().item(), prob.max().item()