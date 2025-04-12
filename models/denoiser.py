# Implements the denoising network
# denoiser can be parametrized as follwos:
# D_theta( x_t, sigma_t) = c_skip(sigma_t) * x_t + c_out(simga_t) * F_theta ( c_in (sigma_t) *x; c_noise (sigma_t) )
# Here F_theta is the raw neural network
# Int he paper there were three architectures specified
# DDPM++, NCSN++ DDPM
# FOR OUR WORK, we will only use DDPM and NCSN
# I am only implementing NCSN, as Van is implementing DDPM
import numpy as np 
import torch
import torch.nn as nn

class NetworkFactory:
    """A factory class that will generate a network acrhictecture, based on the hyperparams specified in the paper"""
    archietctures_ = {"NCSN": dict(c_skip=1, c_out=lambda sigma_t: sigma_t,
                                   c_in=1, c_noise=lambda sigma_t: np.log(0.5*sigma_t)),
                      "edm": dict(c_skip=lambda sigma_t, sigma_data: np.pow(sigma_data, 2)/(np.pow(sigma_data, 2) + np.pow(sigma_t, 2)),
                                    c_out=lambda sigma_t, sigma_data: sigma_t*sigma_data/np.sqrt(np.pow(sigma_data, 2) + np.pow(sigma_t, 2)),
                                    c_in=lambda sigma_t, sigma_data: 1/np.sqrt(np.pow(sigma_data, 2) + np.pow(sigma_t, 2)),
                                    c_noise=lambda sigma_t: np.log(0.25*sigma_t))}


class NCSN(nn.Module):
    """NCSN uses the Unet backbone (same with DDPM), and both were propsoed by Song et.al
    See Table 8, for the network details
    DDPM++:
        Resampling filter: Box
        Noise EMbedding: Positional
        Skip conn encoder: -
        Skip conn decoder: -
        Residual blocks per res: 4
        Attention res: 16
        Attention heads: 1
        Attention blocks in encoder: 4
        Attention blocks in decoder: 2
    NCSN++:
        Resampling filter: Bilinear
        Noise EMbedding: Fourier
        Skip conn encoder: Residual
        Skip conn decoder: -
        Residual blocks per res: 4
        Attention res: 16
        Attention heads: 1
        Attention blocks in encoder: 4
        Attention blocks in decoder: 2
    
    """
    def __init__(self):
        super().__init__()
        self.bilinear_filter_dims  = self.register_buffer("filter_dims", torch.Tensor(1, 3, 3, 1))
        