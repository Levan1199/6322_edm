from models.ncsn_pp import VEPrecond
from data_utils.data_loader import ImageFolderDataset
import os 
from data_utils.data_loader import ImageFolderDataset, InfiniteSampler
import os
from torch.utils.data import DataLoader
import torch 
import torch.optim as optim 
from loss_functions.VE_loss import VELoss
from torch.utils.tensorboard.writer import SummaryWriter
import copy 
import psutil

if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    curr_dir = os.path.dirname(__file__)
    
    path = f"{curr_dir}/../datasets/cifar10-32x32.zip"
    batch_size = 128
    torch.multiprocessing.set_start_method('spawn')
    ds = ImageFolderDataset(path=path, use_labels=False, cache=True)
    sampler = InfiniteSampler(ds)
    iterator = iter(DataLoader(dataset=ds, sampler=sampler, pin_memory=True, num_workers=1, prefetch_factor=2, batch_size=batch_size))
    net_ve = VEPrecond()
    net_ve = net_ve.train().requires_grad_(True).to(device=device)
    optimizer = optim.Adam( net_ve.parameters(),lr=1e-3, betas=[0.9,0.999], eps=1e-8)
    images, labels = next(iterator)
    images = images.to(device).to(torch.float32) / 127.5 - 1
    loss = loss_fn(net=net_ve, images=images)
    logger = SummaryWriter('./ncsnpp')
    ema = copy.deepcopy(net_ve).eval().requires_grad_(False)
    # exit()
    train_iter = 0
    train_steps =0 
    # parameters for noise 
    # described in appendix
    sigma_min = 0.02
    sigma_max = 100
    while True:
    # Accumulate gradients.
        optimizer.zero_grad(set_to_none=True)
        for round_idx in range(1):
            
            images, labels = next(iterator)
            images = images.to(device).to(torch.float32) / 127.5 - 1
            labels = labels.to(device)
            rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
            sigma = sigma_min * ((sigma_max / sigma_min) ** rnd_uniform)
            weight = 1 / sigma ** 2
            y= images
            n = torch.randn_like(y) * sigma
            D_yn = net_ve(y + n, sigma)
            loss = weight * ((D_yn - y) ** 2)
            loss = loss.sum()
            logger.add_scalar('Loss/loss', loss, train_steps)
            loss.backward()

        # Update weights.
        # for g in optimizer.param_groups:
        #     g['lr'] = optimizer_kwargs['lr'] * min(cur_nimg / max(lr_rampup_kimg * 1000, 1e-8), 1)
        # for param in net.parameters():
        #     if param.grad is not None:
        #         torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
        optimizer.step()

        # # Update EMA.
        # ema_halflife_nimg = 500 * 1000
        # if ema_rampup_ratio is not None:
        #     ema_halflife_nimg = min(ema_halflife_nimg, cur_nimg * ema_rampup_ratio)
        # ema_beta = 0.5 ** (batch_size / max(ema_halflife_nimg, 1e-8))
        # for p_ema, p_net in zip(ema.parameters(), net.parameters()):
        #     p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))

        # Perform maintenance tasks once per tick.
        train_iter += batch_size
        train_steps += 1
        done = (train_iter >= int(2e5) * 1000)
        
        if train_steps%int(18e3) == 0:
            torch.save(dict(net=net_ve, optimizer_state=optimizer.state_dict()), os.path.join("./ncsnpp", f'training-state-{train_iter//1000:06d}.chkpt'))
            logger.add_scalar('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30, train_steps)
            logger.add_scalar('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30, train_steps)
            logger.add_scalar('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30, train_steps)
            torch.cuda.reset_peak_memory_stats()
        if (not done):
            continue
        if done:
            break
    torch.save(dict(net=net_ve, optimizer_state=optimizer.state_dict()), os.path.join("./ncsn", f'training-state-{train_iter//1000:06d}.chkpt'))
