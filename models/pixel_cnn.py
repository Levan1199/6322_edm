# pixel cnn++ network architecture
# this is the basic architecture, that is the basis for diffusion model architecture
import torch
import torch.nn as nn

class BaseUnit(nn.Module):
    """This is the base unit for the PixelCNN++
    Adapted from: https://github.com/singh-hrituraj/PixelCNN-Pytorch/tree/master"""
    def __init__(self, mask_type, *args, **kwargs):
        self.mask_type = mask_type
        assert mask_type in ['A', 'B'], "Unknown Mask Type"
        super().__init__(*args, **kwargs)
        self.register_buffer('mask', self.weight.data.clone())

        _, depth, height, width = self.weight.size()
        self.mask.fill_(1)
        if mask_type =='A':
            self.mask[:,:,height//2,width//2:] = 0
            self.mask[:,:,height//2+1:,:] = 0
        else:
            self.mask[:,:,height//2,width//2+1:] = 0
            self.mask[:,:,height//2+1:,:] = 0


    def forward(self, x):
        self.weight.data*=self.mask
        return super().forward(x)

class PixelCNN(nn.Module):
    """PixelCNN ++, this arhitecture will be used for the diffusion model
    For details see: PIXELCNN++: IMPROVING THE PIXELCNN WITH
            DISCRETIZED LOGISTIC MIXTURE LIKELIHOOD AND
            OTHER MODIFICATIONS
        Adapted from: https://github.com/singh-hrituraj/PixelCNN-Pytorch/tree/master"""
    def __init__(self, no_layers=8, kernel = 7, channels=64, device=None):
        super().__init__()
        self.no_layers = no_layers
        self.kernel = kernel
        self.channels = channels
        self.layers = {}
        self.device = device

        self.Conv2d_1 = BaseUnit('A',1,channels, kernel, 1, kernel//2, bias=False)
        self.BatchNorm2d_1 = nn.BatchNorm2d(channels)
        self.ReLU_1= nn.ReLU(True)

        self.Conv2d_2 = BaseUnit('B',channels,channels, kernel, 1, kernel//2, bias=False)
        self.BatchNorm2d_2 = nn.BatchNorm2d(channels)
        self.ReLU_2= nn.ReLU(True)

        self.Conv2d_3 = BaseUnit('B',channels,channels, kernel, 1, kernel//2, bias=False)
        self.BatchNorm2d_3 = nn.BatchNorm2d(channels)
        self.ReLU_3= nn.ReLU(True)

        self.Conv2d_4 = BaseUnit('B',channels,channels, kernel, 1, kernel//2, bias=False)
        self.BatchNorm2d_4 = nn.BatchNorm2d(channels)
        self.ReLU_4= nn.ReLU(True)

        self.Conv2d_5 = BaseUnit('B',channels,channels, kernel, 1, kernel//2, bias=False)
        self.BatchNorm2d_5 = nn.BatchNorm2d(channels)
        self.ReLU_5= nn.ReLU(True)

        self.Conv2d_6 = BaseUnit('B',channels,channels, kernel, 1, kernel//2, bias=False)
        self.BatchNorm2d_6 = nn.BatchNorm2d(channels)
        self.ReLU_6= nn.ReLU(True)

        self.Conv2d_7 = BaseUnit('B',channels,channels, kernel, 1, kernel//2, bias=False)
        self.BatchNorm2d_7 = nn.BatchNorm2d(channels)
        self.ReLU_7= nn.ReLU(True)

        self.Conv2d_8 = BaseUnit('B',channels,channels, kernel, 1, kernel//2, bias=False)
        self.BatchNorm2d_8 = nn.BatchNorm2d(channels)
        self.ReLU_8= nn.ReLU(True)

        self.out = nn.Conv2d(channels, 256, 1)

    def forward(self, x):
        x = self.Conv2d_1(x)
        x = self.BatchNorm2d_1(x)
        x = self.ReLU_1(x)

        x = self.Conv2d_2(x)
        x = self.BatchNorm2d_2(x)
        x = self.ReLU_2(x)

        x = self.Conv2d_3(x)
        x = self.BatchNorm2d_3(x)
        x = self.ReLU_3(x)

        x = self.Conv2d_4(x)
        x = self.BatchNorm2d_4(x)
        x = self.ReLU_4(x)

        x = self.Conv2d_5(x)
        x = self.BatchNorm2d_5(x)
        x = self.ReLU_5(x)

        x = self.Conv2d_6(x)
        x = self.BatchNorm2d_6(x)
        x = self.ReLU_6(x)

        x = self.Conv2d_7(x)
        x = self.BatchNorm2d_7(x)
        x = self.ReLU_7(x)

        x = self.Conv2d_8(x)
        x = self.BatchNorm2d_8(x)
        x = self.ReLU_8(x)

        return self.out(x)