import argparse
import os
import sys
import torch
import torchio as tio
from utils import SubjectsDataset, Trainer, dice_loss, get_dataset_statistics, is_device_valid
from models import *


def is_args_valid(opt):
    if opt.model not in MODELS:
        print(f"Error: {opt.model} is not a valid model, must choose from {MODELS}")
        return False

    if not os.path.exists(opt.dataset_path):
        print(f"Error: dataset path {opt.dataset_path} not found")
        return False

    if not os.path.exists(f"{opt.dataset_path}/train"):
        print(f"Error: training dataset {opt.dataset_path}/train not found")
        return False

    if not os.path.exists(f"{opt.dataset_path}/val"):
        print(f"Error: training dataset {opt.dataset_path}/val not found")
        return False

    if not is_device_valid(opt.device):
        return False

    if opt.unet_initial_channels <= 0:
        print("Error: unet_initial_channels must be positive")
        return False

    if opt.batch_size <= 0:
        print("Error: batch_size must be positive")
        return False

    if opt.num_classes <= 1:
        print("Error: num_classes must be greater than 1")
        return False

    if opt.num_epochs < 0 and opt.patience < 0:
        print("Error: num_epochs or patience must be nonnegative")
        return False

    if opt.resume:
        if opt.model_name is None:
            print(f"Error: model_name must be provided if resuming")
        if not os.path.exists(f"data/{opt.model_name}/weights.pth"):
            print(f"Error: cannot resume from {opt.model_name}. No weights found for {opt.model_name}")

    return True


def train(opt):
    torch.manual_seed(0)
    device = torch.device(opt.device)

    train_path = f"{opt.dataset_path}/train"
    val_path = f"{opt.dataset_path}/val"
    train_dataset = SubjectsDataset(root=train_path)
    val_dataset = SubjectsDataset(root=val_path)
    
    stats = get_dataset_statistics(train_dataset)
    
    train_dataset.set_transform(tio.ZNormalization(masking_method='mask'))
    val_dataset.set_transform(tio.ZNormalization(masking_method=None))

    model_class = getattr(sys.modules[__name__], opt.model)
    if opt.model == "UNet":
        model = model_class(opt.num_classes, opt.unet_initial_channels, device, model_name=opt.model_name)
    else:
        model = model_class(opt.num_classes, device, model_name=opt.model_name)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        loss_fn=dice_loss,
        optimizer=optimizer,
        scheduler=scheduler,
        patience=opt.patience,
        batch_size=opt.batch_size,
        num_classes=opt.num_classes,
        dataset_stats=stats,
        device=device
    )
    
    trainer.train(num_epochs=opt.num_epochs, verbal=opt.verbal)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--model', type=str, default='BaselineModel', help='name of model')
    parser.add_argument('--model_name', type=str, default=None, help='name that weights will be saved to (default is model name)')
    parser.add_argument('--unet_initial_channels', type=int, default=64, help='initial channels for unet model')
    parser.add_argument('--dataset_path', type=str, help='path to dataset', required=True)
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--num_classes', type=int, help='batch size', required=True)
    parser.add_argument('--num_epochs', type=int, default=-1, help='number of epochs (-1 for unlimited)')
    parser.add_argument('--patience', type=int, default=5, help='patience threshold for early stopping (-1 for no early stopping)')
    parser.add_argument('--verbal', type=bool, default=True, help='determines if log is printed')
    parser.add_argument('--resume', action='store_true')  

    opt = parser.parse_args()

    if not is_args_valid(opt):
        sys.exit(1)

    train(opt)
