import torch
import torchio as tio
from .utilFunctions import combine_fragments
from torchmetrics.segmentation import GeneralizedDiceScore


class Evaluator:

    
    def __init__(self, model, dataset, batch_size, num_classes, device):
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.device = device
        self.dice = GeneralizedDiceScore(num_classes=num_classes).to(device)

    
    def iou(self, x, y):
        intersection = torch.logical_and(x, y).sum().item()
        union = torch.logical_or(x, y).sum().item()
        iou = intersection / union
        return iou

    
    def evaluate(self):
        self.model.eval()
        total_dice = torch.zeros(1, device=self.device)
        total_iou = torch.zeros(1, device=self.device)
        
        with torch.no_grad():
            for subject in self.dataset:
                pred = self.model.inference(subject, batch_size=self.batch_size)
                mask = subject['mask'][tio.DATA].to(device=self.device)
                mask = combine_fragments(mask)

                pred = pred.unsqueeze(0).long()
                mask = mask.unsqueeze(0).long()

                total_dice += self.dice(pred, mask)
                total_iou += self.iou(pred, mask)

        avg_dice = total_dice / len(self.dataset)
        avg_iou = total_iou / len(self.dataset)
        
        return avg_dice.item(), avg_iou.item()
