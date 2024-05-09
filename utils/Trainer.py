import itertools
import os
import pickle
import torch
import torch.nn.functional as F
import torchio as tio
from .utilFunctions import combineFragments
from .PatchLoader import PatchLoader
from torchmetrics.segmentation import GeneralizedDiceScore

class Trainer:

    
    def __init__(self, model, train_dataset, val_dataset, loss_fn, optimizer, scheduler, patience, batch_size, device):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.patience = patience
        self.patch_size = self.model.get_patch_size()
        self.batch_size = batch_size
        self.device = device
        self.best_val_loss = float('inf')
        self.counter = 0
        self.best_model_weights = None
        self.losses = []

        self.train_dataloader = PatchLoader(
            train_dataset,
            patch_size=self.patch_size,
            queue_max_length=2048,
            samples_per_volume=2048,
            queue_num_workers=8,
            batch_size=self.batch_size,
        )
        self.val_dataloader = PatchLoader(
            train_dataset,
            patch_size=self.patch_size,
            queue_max_length=2048,
            samples_per_volume=2048,
            queue_num_workers=8,
            batch_size=self.batch_size,
        )

    
    def evaluate(self):
        size = len(self.val_dataset)

        metric = GeneralizedDiceScore(num_classes=4, per_class=True).to(self.device)
    
        self.model.eval()
        total = None
        
        with torch.no_grad():
            for subject in self.val_dataset:
                pred = self.model.inference(subject, batch_size=256)
                mask = subject['mask'][tio.DATA].to(device=self.device)
                mask = combineFragments(mask)
    
                # print(pred.shape)
                # print(mask.shape)
    
                output = metric(pred.long(), mask.long())
    
                if total is None:
                    total = output
                else:
                    total = total.add(output)
    
        
        output /= size
        return output
    
    
    def compute_loss(self, patches_batch):
        mask_patches = patches_batch['mask'][tio.DATA].to(dtype=torch.float, device=self.device)
        mask_patches = combineFragments(mask_patches)
        mask_patches = F.one_hot(mask_patches.squeeze(1).long(), num_classes=4).movedim(-1, 1).float()

        image_patches = patches_batch['image'][tio.DATA].to(device=self.device)
        preds = self.model(image_patches)

        loss = self.loss_fn(preds, mask_patches)
        return loss

    
    def train_epoch(self):
        size = len(self.train_dataloader.dataset)
        self.model.train()

        train_loss = None
        val_loss = None

        for patches_batch in self.train_dataloader:
            self.optimizer.zero_grad()
            loss = self.compute_loss(patches_batch)
            loss.backward()
            self.optimizer.step()

            train_loss = loss if train_loss is None else train_loss.add(loss)

        with torch.no_grad():            
            for patches_batch in self.val_dataloader:
                loss = self.compute_loss(patches_batch)
                val_loss = loss if val_loss is None else val_loss.add(loss)

        

        return train_loss.item(), val_loss.item()

    
    def train(self, num_epochs, model_name=None, verbal=True):

        if num_epochs == -1:
            epoch_iterator = itertools.count()
        else:
            epoch_iterator = range(num_epochs)

        for epoch_num in epoch_iterator:

            train_loss, val_loss = self.train_epoch()
            self.losses.append((train_loss, val_loss))

            self.scheduler.step()

            if verbal:
                print(f"Epoch: {epoch_num:>3d}, Train Loss: {train_loss:>7f}, Val Loss: {val_loss:>7f}")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.counter = 0
                self.best_model_weights = self.model.state_dict()
            else:
                self.counter += 1

            if self.patience < 0 and self.counter >= self.patience:
                print("Early stopping triggered.")
                break

        if self.best_model_weights is not None:

            if model_name is None:
                model_name = self.model.__class__.__name__

            if not os.path.exists(f"data/{model_name}"): 
                os.makedirs(f"data/{model_name}")

            weights_path = f"data/{model_name}/weights.pth"
            torch.save(self.best_model_weights, weights_path)
            
            losses_path = f"data/{model_name}losses.pkl"
            with open(losses_path, 'wb') as f:
                pickle.dump(self.losses, f)

        return self.losses
