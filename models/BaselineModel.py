import torch
import torch.nn as nn
from .PatchModel import PatchModel


class BaselineModel(PatchModel):

    
    def __init__(self, device):
        super().__init__(patch_size=16, device=device)
        self.downsample = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2)
        ).to(device)
        self.upsample = nn.Sequential(
            nn.ConvTranspose3d(16, 16, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv3d(16, 4, kernel_size=1),
            nn.Softmax(dim=1)
        ).to(device)
    
    
    def forward(self, x):
        x = self.downsample(x)
        x = self.upsample(x)
        
        return x
