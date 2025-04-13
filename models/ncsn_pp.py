# Implements the NCSNNpp architecture

import numpy as np
import torch
import torch.nn as nn 
from .unet import (NCSNppUnet, Conv2d)
import torch.nn.functional as func

class CIFAR10_config_VE:
    """A config class that holds CIFAR-10 specific params for VE"""
    img_resolution = 32 # the resolution of input and output images 
    in_channels = 3
    out_channels = 3
    use_fp16        = False
    # VE config specific params
    sigma_min       = 0.02
    sigma_max       = 100
    # NCSN ++ specific params
    channel_multiplier_embedding = 4
    channel_multiplier_noise = 2
    resample_filter = [1,3,3,1]
    feature_channels = 128
    channel_multipliers = [2,2,2]
    label_dim       = 0 # we are only training unconditional
    residual_blocks_per_res = 4 
    # embedding_type = "fourier"
    # encoder = "residual"
    # decoder = "standard"
    

class FourierEmbedding(torch.nn.Module):
    """Adapted from https://github.com/NVlabs/edm/blob/main/training/networks.py
    This represents the fourier embedding type, an alternative to positional embedding.
    Foruier embedding is one of the key differences between DDPM++ and NCSN++"""
    def __init__(self, num_channels, scale=16):
        super().__init__()
        self.register_buffer('freqs', torch.randn(num_channels // 2) * scale)

    def forward(self, x):
        x = x.ger((2 * np.pi * self.freqs).to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x

class VEPrecond(torch.nn.Module):
    """Adapted from  https://github.com/NVlabs/edm/blob/main/training/networks.py
    This implements the Variance Exploding preconditioning as specified in the paper"""
    def __init__(self):
        super().__init__()
        self.config = CIFAR10_config_VE
        self.img_resolution = self.config.img_resolution
        self.img_channels = self.config.in_channels
        self.label_dim = self.config.label_dim
        self.use_fp16 = self.config.use_fp16
        self.sigma_min = self.config.sigma_min
        self.sigma_max = self.config.sigma_max
        self.model = NSCNUnetpp(img_resolution=self.img_resolution,
                                in_channels=self.img_channels, out_channels=self.img_channels,
                                label_dim=self.label_dim)

    def forward(self, x, sigma, class_labels=None, force_fp32=False, **model_kwargs):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32
        
        c_skip = 1
        c_out = sigma
        c_in = 1
        c_noise = (0.5 * sigma).log()

        F_x = self.model((c_in * x).to(dtype), c_noise.flatten(), **model_kwargs)
        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)
    
"""This is customzied for the CIFAR-10 dataset"""
class NSCNUnetpp(torch.nn.Module):
    """Partially adapted from https://github.com/NVlabs/edm/blob/main/training/networks.py"""
    def __init__(self,
        img_resolution,                     
        in_channels,                        
        out_channels,                       
        attention_resolutions    = [16],     # the feature resolutions at which we employ attention mechanism
        dropout             = 0.10,         
        
    ):
      
        super().__init__()
        config = CIFAR10_config_VE()
        feature_channels = config.feature_channels # the number of channels for hidden features
        embedding_channels = feature_channels *config.channel_multiplier_embedding
        noise_channels = feature_channels * config.channel_multiplier_noise
        resample_filter = config.resample_filter
        linear_layer_initialization_kwargs = dict(init_mode='xavier_uniform')
        init_weight_kwargs = dict(init_mode='xavier_uniform', init_weight=1e-5)
        attention_layer_initialization_kwargs = dict(init_mode='xavier_uniform', init_weight=np.sqrt(0.2))

        self.map_noise =  FourierEmbedding(num_channels=noise_channels)
        # self.map_augment = Linear(in_features=augment_dim, out_features=noise_channels, bias=False, **init) if augment_dim else None
        self.map_layer0 = nn.Linear(in_features=noise_channels, out_features=embedding_channels, bias=True)
        self.map_layer1 = nn.Linear(in_features=embedding_channels, out_features=embedding_channels, bias=True)

        # Encoder.
        self.enc = torch.nn.ModuleDict()
        cout = in_channels
        c_auxilliary = in_channels
        for depth, feature_multiplier in enumerate(config.channel_multipliers):
            res = img_resolution // depth
            if depth == 0:
                cin = cout
                cout = feature_channels
                self.enc[f'{res}x{res}_conv'] = nn.Conv2d(in_channels=cin, out_channels=cout, kernel_size=3)
            else:
                self.enc[f'{res}x{res}_down'] = NCSNppUnet(in_channels=cout, out_channels=cout, down=True,  embedding_channels=embedding_channels, num_heads=1, dropout=dropout, skip_scale=np.sqrt(0.5), eps=1e-6,
                                    resample_filter=resample_filter, resample_proj=True, adaptive_scale=False,
                                    init=linear_layer_initialization_kwargs, init_zero=init_weight_kwargs, init_attn=attention_layer_initialization_kwargs)
                
                self.enc[f'{res}x{res}_aux_residual'] = Conv2d(in_channels=c_auxilliary, out_channels=cout, kernel=3, down=True, resample_filter=resample_filter, fused_resample=True)
                c_auxilliary = cout
            for idx in range(config.residual_blocks_per_res):
                cin = cout
                cout = feature_channels * feature_multiplier
                attn = (res in attention_resolutions)
                self.enc[f'{res}x{res}_block{idx}'] = NCSNppUnet(in_channels=cin, out_channels=cout, attention=attn, embedding_channels=embedding_channels, num_heads=1, dropout=dropout, skip_scale=np.sqrt(0.5), eps=1e-6,
                                    resample_filter=resample_filter, resample_proj=True, adaptive_scale=False,
                                    init=linear_layer_initialization_kwargs, init_zero=init_weight_kwargs, init_attn=attention_layer_initialization_kwargs)
        skips = [block.out_channels for name, block in self.enc.items() if 'aux' not in name]

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        for depth, feature_multiplier in reversed(list(enumerate(config.channel_multipliers))):
            res = img_resolution // depth
            if depth == len(config.channel_multipliers) - 1:
                self.dec[f'{res}x{res}_in0'] = NCSNppUnet(in_channels=cout, out_channels=cout, attention=True,  embedding_channels=embedding_channels, num_heads=1, dropout=dropout, skip_scale=np.sqrt(0.5), eps=1e-6,
                                    resample_filter=resample_filter, resample_proj=True, adaptive_scale=False,
                                    init=linear_layer_initialization_kwargs, init_zero=init_weight_kwargs, init_attn=attention_layer_initialization_kwargs)
                self.dec[f'{res}x{res}_in1'] = NCSNppUnet(in_channels=cout, out_channels=cout,  embedding_channels=embedding_channels, num_heads=1, dropout=dropout, skip_scale=np.sqrt(0.5), eps=1e-6,
                                    resample_filter=resample_filter, resample_proj=True, adaptive_scale=False,
                                    init=linear_layer_initialization_kwargs, init_zero=init_weight_kwargs, init_attn=attention_layer_initialization_kwargs)
            else:
                self.dec[f'{res}x{res}_up'] = NCSNppUnet(in_channels=cout, out_channels=cout, up=True,  embedding_channels=embedding_channels, num_heads=1, dropout=dropout, skip_scale=np.sqrt(0.5), eps=1e-6,
                                    resample_filter=resample_filter, resample_proj=True, adaptive_scale=False,
                                    init=linear_layer_initialization_kwargs, init_zero=init_weight_kwargs, init_attn=attention_layer_initialization_kwargs)
            for idx in range(config.residual_blocks_per_res + 1):
                cin = cout + skips.pop()
                cout = feature_channels * feature_multiplier
                attn = (idx == config.residual_blocks_per_res and res in attention_resolutions)
                self.dec[f'{res}x{res}_block{idx}'] = NCSNppUnet(in_channels=cin, out_channels=cout, attention=attn,  embedding_channels=embedding_channels, num_heads=1, dropout=dropout, skip_scale=np.sqrt(0.5), eps=1e-6,
                                    resample_filter=resample_filter, resample_proj=True, adaptive_scale=False,
                                    init=linear_layer_initialization_kwargs, init_zero=init_weight_kwargs, init_attn=attention_layer_initialization_kwargs)
            if depth == 0:
                self.dec[f'{res}x{res}_aux_norm'] = nn.GroupNorm(num_channels=cout, eps=1e-6)
                self.dec[f'{res}x{res}_aux_conv'] = Conv2d(in_channels=cout, out_channels=out_channels, kernel=3)

    def forward(self, x, noise_labels,  augment_labels=None):
        # Mapping.
        emb = self.map_noise(noise_labels)
        emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape) # swap sin/cos
        emb = func.silu(self.map_layer0(emb))
        emb = func.silu(self.map_layer1(emb))

        # Encoder.
        skips = []
        aux = x
        for name, block in self.enc.items():
            # if 'aux_down' in name:
            #     aux = block(aux)
            # elif 'aux_skip' in name:
            #     x = skips[-1] = x + block(aux)
            if 'aux_residual' in name:
                x = skips[-1] = aux = (x + block(aux)) / np.sqrt(2)
            else:
                x = block(x, emb) if isinstance(block, NCSNppUnet) else block(x)
                skips.append(x)

        # Decoder.
        aux = None
        tmp = None
        for name, block in self.dec.items():
            if 'aux_up' in name:
                aux = block(aux)
            elif 'aux_norm' in name:
                tmp = block(x)
            elif 'aux_conv' in name:
                tmp = block(func.silu(tmp))
                aux = tmp if aux is None else tmp + aux
            else:
                if x.shape[1] != block.in_channels:
                    x = torch.cat([x, skips.pop()], dim=1)
                x = block(x, emb)
        return aux
    
############################################################################################################