import torch
import torchio as tio


class PatchLoader(torch.utils.data.DataLoader):

    
    def __init__(self, queue_dataset, patch_size, queue_max_length, samples_per_volume, queue_num_workers, batch_size):
        
        self.queue_dataset = queue_dataset
        self.batch_size = batch_size
        self.queue_max_length = queue_max_length
        self.samples_per_volume = samples_per_volume
        self.queue_num_workers = queue_num_workers
        self.patch_size = patch_size
            
        self.train_queue = tio.Queue(
            queue_dataset,
            max_length=queue_max_length,
            samples_per_volume=samples_per_volume,
            sampler=tio.data.UniformSampler(patch_size=patch_size),
            num_workers=queue_num_workers
        )
        
        super().__init__(self.train_queue, batch_size=batch_size, num_workers=0)
