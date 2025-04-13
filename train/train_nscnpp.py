from models.ncsn_pp import VEPrecond
from data_utils.data_loader import ImageFolderDataset
import os 
from data_utils.data_loader import ImageFolderDataset, InfiniteSampler
import os
from torch.utils.data import DataLoader
import torch 
import torch.optim as optim 
from loss_functions.VE_loss import VELoss

if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    path = "/home/sdcnlab/Desktop/RGS/DL_Course_Project/6322_edm/datasets/cifar10-32x32.zip"
    torch.multiprocessing.set_start_method('spawn')
    ds = ImageFolderDataset(path=path, use_labels=False, cache=True)
    sampler = InfiniteSampler(ds)
    iterator = iter(DataLoader(dataset=ds, sampler=sampler, pin_memory=True, num_workers=1, prefetch_factor=2, batch_size=8))
    net_ve = VEPrecond()
    net_ve = net_ve.train().requires_grad_(True).to(device=device)
    optimizer = optim.Adam( net_ve.parameters(),lr=1e-3, betas=[0.9,0.999], eps=1e-8)
    loss_fn = VELoss()
    images, labels = next(iterator)
    images = images.to(device).to(torch.float32) / 127.5 - 1
    loss = loss_fn(net=net_ve, images=images, labels=labels, augment_pipe=None)
    breakpoint()