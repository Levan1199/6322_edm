"""Most of the acrhitectures, starting from Pixelcnn, DDPM, NCSN++ use UNet backbone"""
import torch 
import torch.nn as nn
import torch.nn.functional as func
import numpy as np 

class Upsampler(nn.Module):
    """Implements the upsampler for Unet, with a FIR filter"""
    def __init__(self, in_channels, out_channels, kernel_size, bias, filter_):
        # NOTE: the unsqueezing may be wrong
        super().__init__()
        filter_ = torch.Tensor(filter_).float32().unsqueeze(0).unsqueeze(0).tile(in_channels, 1, 1, 1)
        self.register_buffer("filter", filter_)
        self.weight_pad  = out_channels//2
        self.filter_pad = filter_.shape[-1] // 2
        # we need kaiming uniform init
        self.convnet = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias, padding=self.weight_pad)
        # assign weights usign kaiming normal
        # note that we cannot use torchs builtin initializer for bias, so we resort to this
        # the fan ins
        fan_in = in_channels*kernel_size*kernel_size
        fan_out = out_channels*kernel_size*kernel_size
        weight_shape = self.convnet.weight.shape
        self.convnet.weight = nn.Parameter(np.sqrt(1 / fan_in) * (torch.rand(*weight_shape)))
        if bias:
            self.convnet.bias = nn.Parameter(np.sqrt(1 / fan_in) * (torch.rand(*self.convnet.bias.shape)))
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        

    def forward(self, x):
        """upsamples, then performs conv"""
        x = func.conv_transpose2d(x, self.filter, groups=self.in_channels, stride=2, padding=self.filter_pad)
        x = self.convnet(x)
        return x
        
class Downsampler(nn.Module):
    """Implements the downsampler for Unet, with a FIR filter"""
    def __init__(self, in_channels, out_channels, kernel_size, bias, filter_):
        # NOTE: the unsqueezing may be wrong
        super().__init__()
        filter_ = torch.Tensor(filter_).float32().unsqueeze(0).unsqueeze(0).tile(in_channels, 1, 1, 1)/4.0
        self.register_buffer("filter", filter_)
        self.weight_pad  = out_channels//2
        self.filter_pad = filter_.shape[-1] // 2
        # we need kaiming uniform init
        self.convnet = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias, padding=self.weight_pad)
        # assign weights usign kaiming normal
        # note that we cannot use torchs builtin initializer for bias, so we resort to this
        # the fan ins
        fan_in = in_channels*kernel_size*kernel_size
        fan_out = out_channels*kernel_size*kernel_size
        weight_shape = self.convnet.weight.shape
        self.convnet.weight = nn.Parameter(np.sqrt(1 / fan_in) * (torch.rand(*weight_shape)))
        if bias:
            self.convnet.bias = nn.Parameter(np.sqrt(1 / fan_in) * (torch.rand(*self.convnet.bias.shape)))
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
def weight_init(shape, mode, fan_in, fan_out):
    """Adapted from https://github.com/NVlabs/edm/blob/main/training/networks.py"""
    if mode == 'xavier_uniform': return np.sqrt(6 / (fan_in + fan_out)) * (torch.rand(*shape) * 2 - 1)
    if mode == 'xavier_normal':  return np.sqrt(2 / (fan_in + fan_out)) * torch.randn(*shape)
    if mode == 'kaiming_uniform': return np.sqrt(3 / fan_in) * (torch.rand(*shape) * 2 - 1)
    if mode == 'kaiming_normal':  return np.sqrt(1 / fan_in) * torch.randn(*shape)
    raise ValueError(f'Invalid init mode "{mode}"')

class Linear(nn.Module):
    """Adapted from https://github.com/NVlabs/edm/blob/main/training/networks.py"""
    def __init__(self, in_features, out_features, bias=True, init_mode='kaiming_normal', init_weight=1, init_bias=0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        init_kwargs = dict(mode=init_mode, fan_in=in_features, fan_out=out_features)
        self.weight = nn.Parameter(weight_init([out_features, in_features], **init_kwargs) * init_weight)
        self.bias = nn.Parameter(weight_init([out_features], **init_kwargs) * init_bias) if bias else None

    def forward(self, x):
        x = x @ self.weight.to(x.dtype).t()
        if self.bias is not None:
            x = x.add_(self.bias.to(x.dtype))
        return x
    
class Conv2d(nn.Module):
    """Adapted from https://github.com/NVlabs/edm/blob/main/training/networks.py"""
    def __init__(self,
        in_channels, out_channels, kernel, bias=True, up=False, down=False,
        resample_filter=[1,1], fused_resample=False, init_mode='kaiming_normal', init_weight=1, init_bias=0,
    ):
        assert not (up and down)
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        self.down = down
        self.fused_resample = fused_resample
        init_kwargs = dict(mode=init_mode, fan_in=in_channels*kernel*kernel, fan_out=out_channels*kernel*kernel)
        self.weight = nn.Parameter(weight_init([out_channels, in_channels, kernel, kernel], **init_kwargs) * init_weight) if kernel else None
        self.bias = nn.Parameter(weight_init([out_channels], **init_kwargs) * init_bias) if kernel and bias else None
        f = torch.as_tensor(resample_filter, dtype=torch.float32)
        f = f.ger(f).unsqueeze(0).unsqueeze(1) / f.sum().square()
        self.register_buffer('resample_filter', f if up or down else None)

    def forward(self, x):
        w = self.weight.to(x.dtype) if self.weight is not None else None
        b = self.bias.to(x.dtype) if self.bias is not None else None
        f = self.resample_filter.to(x.dtype) if self.resample_filter is not None else None
        w_pad = w.shape[-1] // 2 if w is not None else 0
        f_pad = (f.shape[-1] - 1) // 2 if f is not None else 0

        if self.fused_resample and self.up and w is not None:
            x = nn.functional.conv_transpose2d(x, f.mul(4).tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=max(f_pad - w_pad, 0))
            x = nn.functional.conv2d(x, w, padding=max(w_pad - f_pad, 0))
        elif self.fused_resample and self.down and w is not None:
            x = nn.functional.conv2d(x, w, padding=w_pad+f_pad)
            x = nn.functional.conv2d(x, f.tile([self.out_channels, 1, 1, 1]), groups=self.out_channels, stride=2)
        else:
            if self.up:
                x = nn.functional.conv_transpose2d(x, f.mul(4).tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=f_pad)
            if self.down:
                x = nn.functional.conv2d(x, f.tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=f_pad)
            if w is not None:
                x = nn.functional.conv2d(x, w, padding=w_pad)
        if b is not None:
            x = x.add_(b.reshape(1, -1, 1, 1))
        return x
    
class AttentionOp(torch.autograd.Function):
    """Adapted from https://github.com/NVlabs/edm/blob/main/training/networks.py"""
    @staticmethod
    def forward(ctx, q, k):
        w = torch.einsum('ncq,nck->nqk', q.to(torch.float32), (k / np.sqrt(k.shape[1])).to(torch.float32)).softmax(dim=2).to(q.dtype)
        ctx.save_for_backward(q, k, w)
        return w

    @staticmethod
    def backward(ctx, dw):
        q, k, w = ctx.saved_tensors
        db = torch._softmax_backward_data(grad_output=dw.to(torch.float32), output=w.to(torch.float32), dim=2, input_dtype=torch.float32)
        dq = torch.einsum('nck,nqk->ncq', k.to(torch.float32), db).to(q.dtype) / np.sqrt(k.shape[1])
        dk = torch.einsum('ncq,nqk->nck', q.to(torch.float32), db).to(k.dtype) / np.sqrt(k.shape[1])
        return dq, dk

class GroupNorm(nn.Module):
    def __init__(self, num_channels, num_groups=32, min_channels_per_group=4, eps=1e-5):
        super().__init__()
        self.num_groups = min(num_groups, num_channels // min_channels_per_group)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        x = nn.functional.group_norm(x, num_groups=self.num_groups, weight=self.weight.to(x.dtype), bias=self.bias.to(x.dtype), eps=self.eps)
        return x


class FourierEmbedding(nn.Module):
    """adapted from: NCSN implemetation 
    @https://github.com/NVlabs/edm/blob/main/training/networks.py"""
    def __init__(self, num_channels, scale=16):
        super().__init__()
        self.register_buffer('freqs', torch.randn(num_channels // 2) * scale)

    def forward(self, x):
        x = x.ger((2 * np.pi * self.freqs).to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x
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
        to StyleGAN-2
        NOTE: We may need to implement kaiming_normal init for GroupNorm and Linear
        Adapted from https://github.com/NVlabs/edm/blob/main/training/networks.py"""
    @staticmethod
    def get_num_groups(num_channels):
        """returns the number of groups givent he number of channels"""
        return min(32, num_channels // 4)
    
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
        self.affine = Linear(in_features=emb_channels, out_features=out_channels*(2 if adaptive_scale else 1))
        self.norm1 = GroupNorm( num_channels=out_channels, eps=eps)
        self.conv1 = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=3, **init_zero)

        self.skip = None
        if out_channels != in_channels or up or down:
            kernel = 1 if resample_proj or out_channels!= in_channels else 0
            self.skip = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel=kernel, up=up, down=down, resample_filter=resample_filter, **init)

        if self.num_heads:
            self.norm2 = GroupNorm( num_channels=out_channels, eps=eps)
            self.qkv = Conv2d(in_channels=out_channels, out_channels=out_channels*3, kernel=1, **(init_attn if init_attn is not None else init))
            self.proj = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=1, **init_zero)

    def forward(self, x, emb):
        orig = x
        x = self.conv0(func.silu(self.norm0(x)))

        params = self.affine(emb).unsqueeze(2).unsqueeze(3).to(x.dtype)
        if self.adaptive_scale:
            scale, shift = params.chunk(chunks=2, dim=1)
            x = func.silu(torch.addcmul(shift, self.norm1(x), scale + 1))
        else:
            x = func.silu(self.norm1(x.add_(params)))

        x = self.conv1(nn.functional.dropout(x, p=self.dropout, training=self.training))
        # breakpoint()
        x = x.add_(self.skip(orig) if self.skip is not None else orig)
        x = x * self.skip_scale

        if self.num_heads:
            q, k, v = self.qkv(self.norm2(x)).reshape(x.shape[0] * self.num_heads, x.shape[1] // self.num_heads, 3, -1).unbind(2)
            w = AttentionOp.apply(q, k)
            a = torch.einsum('nqk,nck->ncq', w, v)
            x = self.proj(a.reshape(*x.shape)).add_(x)
            x = x * self.skip_scale
        return x
