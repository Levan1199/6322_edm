import torch
"""Adapted from https://github.com/NVlabs/edm/blob/main/training/loss.py"""

class VELoss:
    def __init__(self, sigma_min=0.02, sigma_max=100):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, net, images):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma_min * ((self.sigma_max / self.sigma_min) ** rnd_uniform)
        weight = 1 / sigma ** 2
        y= images
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma)
        loss = weight * ((D_yn - y) ** 2)
        return loss
