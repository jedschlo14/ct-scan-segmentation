import os
import pickle
import torch


def compute_dataset_statistics(dataset):
    masks = list(map(lambda x: torch.from_numpy(x['mask'].numpy()).float(), dataset))
    classes = torch.cat([tensor.view(-1) for tensor in masks])
    
    counts = torch.unique(classes, return_counts=True)
    indices = counts[0].logical_and((counts[0] >= 1))
    count = counts[1][indices].sum()
    total_count = counts[1].sum()

    weight = torch.tensor([total_count / (2 * counts[1][0]),  total_count / (2 * count)])
    stats = {'weight': weight}
    
    return stats


def get_dataset_statistics(dataset):

    if not os.path.exists("data"): 
        os.makedirs("data")
    
    pickle_path = "data/datasetStats.pkl"
    
    if os.path.exists(pickle_path):
        with open(pickle_path, 'rb') as f:
            stats = pickle.load(f)
    else:
        stats = compute_dataset_statistics(dataset)
        with open(pickle_path, 'wb') as f:
            pickle.dump(stats, f)
    return stats


def combine_fragments(mask):
    mask = mask.clone()
    mask[mask > 0] = 1
    return mask


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
