import argparse
import sys
import torch
import torchio as tio
from utils import getDatasetStatistics, SubjectsDataset, PatchLoader, Trainer
from models import BaselineModel, VolumetricFCN, UNet, MODELS


def is_device_valid(device):
    if device.lower() == 'cpu':
        return True
        
    elif device.lower() == 'cuda':
        if not torch.cuda.is_available():
            print("Error: CUDA is not available.")
            return False
            
    elif device.lower().startswith('cuda:'):
        device_id = device.lower().replace('cuda:', '')
        
        try:
            device_id = int(device_id)
            if not (0 <= device_id < torch.cuda.device_count()):
                print(f"Error: CUDA device index {device_id} is not valid.")
                return False
                
        except ValueError:
            print("Error: Invalid CUDA device index.")
            return False
            
    else:
        print("Error: Invalid device string.")
        return False

    return True


def is_args_valid(opt):
    if opt.model not in MODELS:
        print(f"Error: {opt.model} is not a valid model, must choose from {MODELS}")
        return False

    if not is_device_valid(opt.device):
        return False

    if opt.batch_size <= 0:
        print("Error: batch_size must be positive")
        return False

    if opt.num_epochs < 0 and opt.patience < 0:
        print("Error: num_epochs or patience must be nonnegative")
        return False

    return True


def train(opt):
    torch.manual_seed(0)
    train_dataset = SubjectsDataset(root='dataset/train')
    val_dataset = SubjectsDataset(root='dataset/val')
    device = torch.device(opt.device)
    stats = getDatasetStatistics(train_dataset)
    
    # train_dataset.set_transform(tio.Compose([
    #     tio.RescaleIntensity((0, 1)),  # Rescale intensities to range [0, 1]
    #     tio.RandomAffine(scales=(0.9, 1.1), degrees=10, isotropic=False),  # Random affine transformation
    #     tio.RandomFlip(axes=(0, 1, 2)),  # Randomly flip along axes
    #     # tio.RandomElasticDeformation(num_control_points=(7, 7, 7), max_displacement=(10, 10, 10)),  # Random elastic deformation
    #     tio.RandomNoise(std=(0, 0.1)),  # Add random noise
    #     # tio.RandomBiasField(coefficients=(0, 0.5)),  # Add random bias field
    #     # tio.RandomMotion(p=0.2),  # Apply random motion artifact
    #     tio.ZNormalization(masking_method='mask')
    # ]))
    
    train_dataset.set_transform(tio.ZNormalization(masking_method='mask'))
    val_dataset.set_transform(tio.ZNormalization(masking_method='mask'))

    model_class = getattr(sys.modules[__name__], opt.model)
    model = model_class(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=stats["weight"], reduction="mean").to(device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        patience=opt.patience,
        batch_size=opt.batch_size,
        device=device
    )
    
    trainer.train(num_epochs=opt.num_epochs, model_name=opt.model_name, verbal=opt.verbal)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--model', type=str, default='BaselineModel', help='name of model')
    parser.add_argument('--model_name', type=str, default=None, help='name that weights will be saved to (default is model name)')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs (-1 for unlimited)')
    parser.add_argument('--patience', type=int, default=5, help='patience threshold for early stopping (-1 for no early stopping)')
    parser.add_argument('--verbal', type=bool, default=True, help='determines if log is printed')
    opt = parser.parse_args()

    if not is_args_valid(opt):
        sys.exit(1)

    train(opt)
