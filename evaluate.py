import pickle
import torch
import torchio as tio
from utils import SubjectsDataset
from models import *


def is_args_valid(opt):
    if opt.model not in MODELS:
        print(f"Error: {opt.model} is not a valid model, must choose from {MODELS}")
        return False

    if not os.path.exists(opt.dataset_path):
        print(f"Error: dataset path {opt.dataset_path} not found")
        return False

    if not os.path.exists(f"{opt.dataset_path}/val"):
        print(f"Error: training dataset {opt.dataset_path}/val not found")
        return False
        
    if not is_device_valid(opt.device):
        return False

    if not os.path.exists(f"data/{opt.model_name}/weights.pth"):
        print(f"Error: no weights found for {opt.model_name}")

    if opt.unet_initial_channels <= 0:
        print("Error: unet_initial_channels must be positive")
        return False
    
    if opt.batch_size <= 0:
        print("Error: batch_size must be positive")
        return False

    if opt.num_classes <= 1:
        print("Error: num_classes must be greater than 1")
        return False

    return True


def evaluate(opt):
    torch.manual_seed(0)
    device = torch.device(opt.device)
    
    dataset = SubjectsDataset(root=f'datasets/{opt.dataset_path}/test')
    dataset.set_transform(tio.ZNormalization(masking_method=None))
    
    model_class = getattr(sys.modules[__name__], opt.model)
    if opt.model == "UNet":
        model = model_class(opt.num_classes, opt.unet_initial_channels, device)
    else:
        model = model_class(opt.num_classes, device)

    model_name = model.__class__.__name__ if opt.model_name is None else opt.model_name
    model.load_state_dict(torch.load(f"data/{model_name}/weights.pth", map_location=device))

    evaluator = Evaluator(
        model=model,
        dataset=dataset,
        batch_size=opt.batch_size,
        num_classes=opt.num_classes,
        device=device
    )

    avg_dice, avg_iou = evaluator.evaluate()

    evaluation = {
        "dice": avg_dice,
        "iou": avg_iou
    }

    if opt.verbal:
        print(f"Average Dice: {avg_dice}")
        print(f"Average IoU: {avg_iou}")

    if not os.path.exists(f"data/{model_name}"): 
        os.makedirs(f"data/{model_name}")

    with open(f"data/{model_name}/eval.pkl", 'wb') as f:
        pickle.dump(evaluation, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--model', type=str, default='BaselineModel', help='name of model')
    parser.add_argument('--model_name', type=str, default=None, help='name that weights will be loaded from (default is model name)')
    parser.add_argument('--dataset_path', type=str, help='path to dataset', required=True)
    parser.add_argument('--unet_initial_channels', type=int, default=64, help='initial channels for unet model')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--num_classes', type=int, help='batch size', required=True)
    parser.add_argument('--verbal', type=bool, default=True, help='determines if eval is printed')
    opt = parser.parse_args()

    if not is_args_valid(opt):
        sys.exit(1)

    evaluate(opt)
    