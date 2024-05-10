import itertools
import os
import pickle
import torch
import torch.nn.functional as F
import torchio as tio
from .utilFunctions import combine_fragments
from .PatchLoader import PatchLoader


class Trainer:

    
    def __init__(self, model, train_dataset, val_dataset, loss_fn, optimizer, scheduler, patience, batch_size, num_classes, dataset_stats, device):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.patience = patience
        self.patch_size = self.model.get_patch_size()
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.dataset_stats = dataset_stats
        self.device = device
        self.best_val_loss = float('inf')
        self.counter = 0
        self.losses = []

        self.train_dataloader = PatchLoader(
            train_dataset,
            patch_size=self.patch_size,
            queue_max_length=256,
            samples_per_volume=64,
            queue_num_workers=8,
            batch_size=self.batch_size,
            dataset_stats = self.dataset_stats
        )
    
    
    def compute_loss(self, patches_batch):
        mask_patches = patches_batch['mask'][tio.DATA].to(dtype=torch.float, device=self.device)
        mask_patches = combine_fragments(mask_patches)
        mask_patches = F.one_hot(mask_patches.squeeze(1).long(), num_classes=self.num_classes).movedim(-1, 1).float()

        image_patches = patches_batch['image'][tio.DATA].to(device=self.device)
        preds = self.model(image_patches)
        
        loss = self.loss_fn(preds, mask_patches)
        return loss

    
    def train_epoch(self):
        size = len(self.train_dataloader.dataset)
        self.model.train()

        train_loss = 0.0
        val_loss = 0.0

        for patches_batch in self.train_dataloader:
            num_patches = patches_batch['image'][tio.DATA].shape[0]
            
            self.optimizer.zero_grad()
            loss = self.compute_loss(patches_batch)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item() / num_patches

        with torch.no_grad():    
            self.model.eval()
            
            for subject in self.val_dataset:

                grid_sampler = tio.inference.GridSampler(
                    subject,
                    patch_size=self.patch_size,
                    patch_overlap=0,
                )
                patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=self.batch_size, shuffle=False)

                for patches_batch in patch_loader:
                    num_patches = patches_batch['image'][tio.DATA].shape[0]
                
                    loss = self.compute_loss(patches_batch)
                    val_loss += loss.item() / num_patches

        return train_loss, val_loss

    
    def train(self, num_epochs, verbal=True):

        if not os.path.exists(f"data/{self.model.name}"): 
            os.makedirs(f"data/{self.model.name}")

        weights_path = f"data/{self.model.name}/weights.pth"
        losses_path = f"data/{self.model.name}/losses.pkl"

        if num_epochs == -1:
            epoch_iterator = itertools.count()
        else:
            epoch_iterator = range(num_epochs)

        for epoch_num in epoch_iterator:

            train_loss, val_loss = self.train_epoch()
            self.losses.append((train_loss, val_loss))

            self.scheduler.step()
            
            if verbal:
                print(f"Epoch: {epoch_num + 1:>3d}, Train Loss: {train_loss:>7f}, Val Loss: {val_loss:>7f}")

            with open(losses_path, 'wb') as f:
                pickle.dump(self.losses, f)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.counter = 0
                torch.save(self.model.state_dict(), weights_path)
            else:
                self.counter += 1

            if self.patience >= 0 and self.counter >= self.patience:
                print("Early stopping triggered.")
                break
            

        return self.losses
