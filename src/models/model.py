# Physics-aware model definition
# Contains neural network architecture with physical constraints

import torch
import torch.nn as nn
from smoke_phys.attention import PhysicsGuidedAttention

class PhysicsAwareNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            PhysicsGuidedAttention(32),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            PhysicsGuidedAttention(64),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        attention_maps = []
        regs = []
        
        for layer in self.feature_extractor:
            if isinstance(layer, PhysicsGuidedAttention):
                x, reg = layer(x)
                regs.append(reg)
            else:
                x = layer(x)
                
        x = self.classifier(x)
        return x, sum(regs)/len(regs) if regs else 0