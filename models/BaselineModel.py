import torch
import torch.nn as nn
from .PatchModel import PatchModel


class BaselineModel(PatchModel):

    
    def __init__(self, device):
        super().__init__(patch_size=16, device=device)
        self.downsample = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        ).to(device)
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 4, kernel_size=1),
            nn.Softmax(dim=1)
        ).to(device)
    
    
    def forward(self, x):
        print(x.shape)
        # x = x.squeeze(
        x = self.downsample(x)
        x = self.upsample(x)
        
        return x
