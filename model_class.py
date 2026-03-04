import torch
import torch.nn as nn
import torchvision.models as models

class GaussianNoise(nn.Module):
    """
    Injects random noise into the 1D feature representations during training.
    This prevents the dense layers from co-adapting to specific, dominant signals.
    """
    def __init__(self, std=0.15):
        super().__init__()
        self.std = std

    def forward(self, x):
        if self.training and self.std > 0:
            noise = torch.randn_like(x) * self.std
            return x + noise
        return x

class MyAgeClassifier(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.5):
        super().__init__()
        
        # CONSTRAINT: Train strictly from scratch
        self.backbone = models.resnet18(weights=None)
        
        # Extract features (512 for ResNet-18)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # Robust decision head
        self.robust_head = nn.Sequential(
            GaussianNoise(std=0.15),
            nn.Dropout(p=dropout_rate),
            nn.Linear(num_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate), # Second dropout for heavy regularization
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        out = self.robust_head(features)
        return out

def build_model(num_classes=2):
    return MyAgeClassifier(num_classes=num_classes)
