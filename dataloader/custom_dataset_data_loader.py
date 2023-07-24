import torch.utils.data
from dataloader.base_data_loader import BaseDataLoader

def CreateDataset(opt):
    if opt.dataset_mode == 'video':
        from dataloader.video_dataset import videoDataset
        dataset = videoDataset()
    else:
        raise ValueError('Unrecognized dataset')
    dataset.initialize(opt)
    return dataset

class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt, start_idx):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)
        self.sampler = MySequentialSampler(self.dataset, start_idx) if opt.serial_batches else None
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=not opt.serial_batches,
            sampler=self.sampler,
            num_workers=int(opt.nThreads),
            drop_last=opt.batch_size>1)

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return len(self.dataset)

class MySequentialSampler(torch.utils.data.Sampler):
    def __init__(self, data_source, start_idx):
        self.data_source = data_source
        self.start_idx = start_idx

    def __iter__(self):
        return iter(range(self.start_idx, len(self.data_source)))

    def __len__(self):
        return len(self.data_source) - self.start_idx
