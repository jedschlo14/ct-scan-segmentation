""" Full assembly of the parts to form the complete network """

from .unet_parts import *
from ..PatchModel import PatchModel

class UNet(PatchModel):
    def __init__(self, device, bilinear=False):
        super().__init__(patch_size=16, device=device)
        
        self.n_channels = 1
        self.n_classes = 4
        self.bilinear = bilinear
        self.initial_channels = 16
        self.inner_channels = [self.initial_channels * 2**i for i in range(5)]

        self.inc = (DoubleConv(self.n_channels, self.inner_channels[0])).to(device)
        self.down1 = (Down(self.inner_channels[0], self.inner_channels[1])).to(device)
        self.down2 = (Down(self.inner_channels[1], self.inner_channels[2])).to(device)
        self.down3 = (Down(self.inner_channels[2], self.inner_channels[3])).to(device)
        factor = 2 if bilinear else 1
        self.down4 = (Down(self.inner_channels[3], self.inner_channels[4] // factor)).to(device)
        self.up1 = (Up(self.inner_channels[4], self.inner_channels[3] // factor, self.bilinear)).to(device)
        self.up2 = (Up(self.inner_channels[3], self.inner_channels[2] // factor, self.bilinear)).to(device)
        self.up3 = (Up(self.inner_channels[2], self.inner_channels[1] // factor, self.bilinear)).to(device)
        self.up4 = (Up(self.inner_channels[1], self.inner_channels[0], self.bilinear)).to(device)
        self.outc = (OutConv(self.inner_channels[0], self.n_classes)).to(device)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)