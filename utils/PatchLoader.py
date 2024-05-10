import torch
import torchio as tio


class PatchLoader(torch.utils.data.DataLoader):

    
    def __init__(self, queue_dataset, patch_size, queue_max_length, samples_per_volume, queue_num_workers, batch_size, dataset_stats):
        
        self.queue_dataset = queue_dataset
        self.batch_size = batch_size
        self.queue_max_length = queue_max_length
        self.samples_per_volume = samples_per_volume
        self.queue_num_workers = queue_num_workers
        self.patch_size = patch_size
        self.dataset_stats = dataset_stats

        self.probabilities = {
            0: dataset_stats["weight"][0]
        }
        for i in range(1, 31):
            self.probabilities[i] = dataset_stats["weight"][1]

        sampler = tio.data.LabelSampler(
            patch_size=patch_size,
            label_name='mask',
            label_probabilities=self.probabilities,
        )
            
        train_queue = tio.Queue(
            queue_dataset,
            max_length=queue_max_length,
            samples_per_volume=samples_per_volume,
            sampler=sampler,
            num_workers=queue_num_workers
        )
        
        super().__init__(train_queue, batch_size=batch_size, num_workers=0)
