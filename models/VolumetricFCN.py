import torch
import torch.nn as nn
from .PatchModel import PatchModel

class VolumetricFCN(PatchModel):

    def __init__(self, num_classes, device):
        super().__init__(patch_size=32, device=device, model_name=None)

        self.conv1 = self.ConvUnit(1, 64).to(device)
        self.conv2 = self.ConvUnit(64, 128).to(device)
        self.conv3 = self.ConvUnit(128, 256).to(device)
        self.conv4 = self.ConvUnit(256, 512).to(device)
        self.conv5 = self.ConvUnit(512, 512).to(device)

        self.conv6 = nn.Sequential(
            nn.Conv3d(512, 4096, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Dropout3d()
        ).to(device)

        self.conv7 = nn.Sequential(
            nn.Conv3d(4096, 4096, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Dropout3d()
        ).to(device)

        self.score = nn.Conv3d(4096, num_classes, kernel_size=1).to(device)
        self.upscore = nn.ConvTranspose3d(num_classes, num_classes, kernel_size=32, stride=1, bias=False).to(device)

        self.name = __class__.__name__ if model_name is None else model_name
    

    def ConvUnit(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.score(x)
        x = self.upscore(x)

        return x
