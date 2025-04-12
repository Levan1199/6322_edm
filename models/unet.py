"""Most of the acrhitectures, starting from Pixelcnn, DDPM, NCSN++ use UNet backbone"""
import torch 
import torch.nn as nn
import torch.nn.functional as func
import numpy as np 

class Upsampler(nn.Module):
    """Implements the upsampler for Unet, with a FIR filter"""
    def __init__(self, in_channels, out_channels, kernel_size, use_bias):
        # NOTE: the unsqueezing may be wrong
        super().__init__()
        filter_ = torch.ones((2,2), dtype=torch.float32).unsqueeze(0).unsqueeze(0).tile(in_channels, 1, 1, 1)
        self.register_buffer("filter", filter_)
        self.weight_pad  = out_channels//2
        self.filter_pad = filter_.shape[-1] // 2
        # we need kaiming uniform init
        self.convnet = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=use_bias, padding=self.weight_pad)
        # assign weights usign kaiming normal
        # note that we cannot use torchs builtin initializer for bias, so we resort to this
        # the fan ins
        fan_in = in_channels*kernel_size*kernel_size
        fan_out = out_channels*kernel_size*kernel_size
        weight_shape = self.convnet.shape
        self.convnet.weight = torch.nn.Parameter(np.sqrt(1 / fan_in) * (torch.rand(*weight_shape)))
        if use_bias:
            self.convnet.bias = torch.nn.Parameter(np.sqrt(1 / fan_in) * (torch.rand(*self.convnet.bias.shape)))
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        

    def forward(self, x):
        """upsamples, then performs conv"""
        x = func.conv_transpose2d(x, self.filter, groups=self.in_channels, stride=2, padding=self.filter_pad)
        x = self.convnet(x)
        return x
        
class Downsampler(nn.Module):
    """Implements the downsampler for Unet, with a FIR filter"""
    def __init__(self, in_channels, out_channels, kernel_size, use_bias):
        # NOTE: the unsqueezing may be wrong
        super().__init__()
        filter_ = torch.ones((2,2), dtype=torch.float32).unsqueeze(0).unsqueeze(0).tile(in_channels, 1, 1, 1)/4.0
        self.register_buffer("filter", filter_)
        self.weight_pad  = out_channels//2
        self.filter_pad = filter_.shape[-1] // 2
        # we need kaiming uniform init
        self.convnet = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=use_bias, padding=self.weight_pad)
        # assign weights usign kaiming normal
        # note that we cannot use torchs builtin initializer for bias, so we resort to this
        # the fan ins
        fan_in = in_channels*kernel_size*kernel_size
        fan_out = out_channels*kernel_size*kernel_size
        weight_shape = self.convnet.shape
        self.convnet.weight = torch.nn.Parameter(np.sqrt(1 / fan_in) * (torch.rand(*weight_shape)))
        if use_bias:
            self.convnet.bias = torch.nn.Parameter(np.sqrt(1 / fan_in) * (torch.rand(*self.convnet.bias.shape)))
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        

    def forward(self, x):
        """upsamples, then performs conv"""
        x = func.conv_2d(x, self.filter, groups=self.in_channels, stride=2, padding=self.filter_pad)
        x = self.convnet(x)
        return x
        
        
class NCSNppUnet(nn.Module):
    """This is the Unet architecture used in NCSN++ paper by Song, et.al. Thier architecture was based on
    Ho, et. al. In addition to Ho, et.als work, Song et. al introduced the following:
        1. Upsampling and downsampling images with anti-aliasing based on Finite Impulse Re-
        sponse (FIR) (Zhang, 2019). We follow the same implementation and hyper-parameters in
        StyleGAN-2 (Karras et al., 2020b).
        2. Rescaling all skip connections by 1{?2. This has been demonstrated effective in several best-
        in-class GAN models, including ProgressiveGAN (Karras et al., 2018), StyleGAN (Karras
        et al., 2019) and StyleGAN-2 (Karras et al., 2020b).
        3. Replacing the original residual blocks in DDPM with residual blocks from BigGAN (Brock
        et al., 2018).
        4. Increasing the number of residual blocks per resolution from 2 to 4.
        5. Incorporating progressive growing architectures. We consider two progressive architectures
        for input: “input skip” and “residual”, and two progressive architectures for output: “output
        skip” and “residual”. These progressive architectures are defined and implemented according
        to StyleGAN-2"""
    def __init__(self,
        in_channels, out_channels, emb_channels, up=False, down=False, attention=False,
        num_heads=None, channels_per_head=64, dropout=0, skip_scale=1, eps=1e-5,
        resample_filter=[1,1], resample_proj=False, adaptive_scale=True,
        init=dict(), init_zero=dict(init_weight=0), init_attn=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_channels = emb_channels
        self.num_heads = 0 if not attention else num_heads if num_heads is not None else out_channels // channels_per_head
        self.dropout = dropout
        self.skip_scale = skip_scale
        self.adaptive_scale = adaptive_scale

        self.norm0 = GroupNorm(num_channels=in_channels, eps=eps)
        self.conv0 = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel=3, up=up, down=down, resample_filter=resample_filter, **init)
        self.affine = Linear(in_features=emb_channels, out_features=out_channels*(2 if adaptive_scale else 1), **init)
        self.norm1 = GroupNorm(num_channels=out_channels, eps=eps)
        self.conv1 = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=3, **init_zero)

        self.skip = None
        if out_channels != in_channels or up or down:
            kernel = 1 if resample_proj or out_channels!= in_channels else 0
            self.skip = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel=kernel, up=up, down=down, resample_filter=resample_filter, **init)

        if self.num_heads:
            self.norm2 = GroupNorm(num_channels=out_channels, eps=eps)
            self.qkv = Conv2d(in_channels=out_channels, out_channels=out_channels*3, kernel=1, **(init_attn if init_attn is not None else init))
            self.proj = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=1, **init_zero)

    def forward(self, x, emb):
        orig = x
        x = self.conv0(silu(self.norm0(x)))

        params = self.affine(emb).unsqueeze(2).unsqueeze(3).to(x.dtype)
        if self.adaptive_scale:
            scale, shift = params.chunk(chunks=2, dim=1)
            x = silu(torch.addcmul(shift, self.norm1(x), scale + 1))
        else:
            x = silu(self.norm1(x.add_(params)))

        x = self.conv1(torch.nn.functional.dropout(x, p=self.dropout, training=self.training))
        x = x.add_(self.skip(orig) if self.skip is not None else orig)
        x = x * self.skip_scale

        if self.num_heads:
            q, k, v = self.qkv(self.norm2(x)).reshape(x.shape[0] * self.num_heads, x.shape[1] // self.num_heads, 3, -1).unbind(2)
            w = AttentionOp.apply(q, k)
            a = torch.einsum('nqk,nck->ncq', w, v)
            x = self.proj(a.reshape(*x.shape)).add_(x)
            x = x * self.skip_scale
        return x

        