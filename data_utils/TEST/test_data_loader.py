from data_utils.data_loader import ImageFolderDataset, InfiniteSampler
import os
from torch.utils.data import DataLoader
import torch 

if __name__ == "__main__":
    path = "/home/sdcnlab/Desktop/RGS/DL_Course_Project/6322_edm/datasets/cifar10-32x32.zip"
    torch.multiprocessing.set_start_method('spawn')
    ds = ImageFolderDataset(path=path, use_labels=False, cache=True)
    sampler = InfiniteSampler(ds)
    iterator = iter(DataLoader(dataset=ds, sampler=sampler, pin_memory=True, num_workers=1, prefetch_factor=2, batch_size=8))
    # pin_memory=True, num_workers=opts.workers, prefetch_factor=2
    breakpoint()