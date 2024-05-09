import torch
import torch.nn as nn
import torchio as tio


class PatchModel(nn.Module):

    
    def __init__(self, patch_size, device):
        super().__init__()
        self.patch_size = patch_size
        self.device = device


    def get_patch_size(self):
        return self.patch_size


    def inference(self, subject, batch_size):
        self.eval()
        
        grid_sampler = tio.inference.GridSampler(
            subject,
            patch_size=self.patch_size,
            patch_overlap=0,
        )
        patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=batch_size)
        aggregator = tio.inference.GridAggregator(grid_sampler)
        
        for patches_batch in patch_loader:
            image_patches = patches_batch['image'][tio.DATA].to(device=self.device)
            locations = patches_batch[tio.LOCATION]
            logits = self(image_patches)
            labels = logits.argmax(dim=tio.CHANNELS_DIMENSION, keepdim=True)
            aggregator.add_batch(labels, locations)
    
        pred = aggregator.get_output_tensor().to(device=self.device)
        return pred