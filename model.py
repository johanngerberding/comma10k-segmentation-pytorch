import numpy as np
import torch 
import torch.nn as nn 
import torch.nn.functional as F 


class RegSeg(nn.Module):
    def __init__(self, encoder, decoder):
        super(RegSeg, self).__init__()
        self.encoder = encoder 
        self.decoder = decoder
        
    def forward(self, x):
        return x
    
    
class Encoder(nn.Module):
    def __init__(self, out_channels=320):
        super(Encoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.d_blocks = nn.Sequential(
            DBlock(32, 48, 1, 1, 2),
            DBlock(48, 48, 1, 1, 2),
            DBlock(48, 48, 1, 1, 1),
            DBlock(48, 128, 1, 1, 1),
            DBlock(128, 128, 1, 1, 2),
            DBlock(128, 256, 1, 1, 1),
            DBlock(256, 256, 1, 2, 1),
            DBlock(256, 256, 1, 4, 1),
            DBlock(256, 256, 1, 4, 1),
            DBlock(256, 256, 1, 4, 1),
            DBlock(256, 256, 1, 4, 1),
            DBlock(256, 256, 1, 14, 1),
            DBlock(256, 256, 1, 14, 1),
            DBlock(256, 256, 1, 14, 1),
            DBlock(256, 256, 1, 14, 1),
            DBlock(256, 256, 1, 14, 1),
            DBlock(256, 256, 1, 14, 1),
            DBlock(256, out_channels, 1, 14, 1),
        )
        
        
    def forward(self, x):
        filter_size = self.conv[0].kernel_size[0]
        pads = get_same_pads(x, filter_size, 2)
        x = F.pad(x, pads)
        x = self.conv(x)
        print(x.size())
        x = self.d_blocks(x)
        return x 
    
    
class Decoder(nn.Module):   
    def __init__(self):
        super(Decoder, self).__init__()
        
    def forward(self, x):
        return x 



class DBlock(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        d1, 
        d2, 
        stride, 
        se_ratio=0.25,
    ):
        super(DBlock, self).__init__()
        assert in_channels % 2 == 0
        self.in_channels = in_channels 
        self.out_channels = out_channels 
        self.group = in_channels // 2
        self.stride = stride
        
        self.conv1 = nn.Conv2d(in_channels, in_channels, 1, 1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(
            self.group, self.group, 3, stride, dilation=d1)
        self.bn2 = nn.BatchNorm2d(self.group)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.conv3 = nn.Conv2d(
            self.group, self.group, 3, stride, dilation=d2)
        self.bn3 = nn.BatchNorm2d(self.group)
        self.relu3 = nn.ReLU(inplace=True)
        
        self.conv4 = nn.Conv2d(in_channels, out_channels, 1, 1)
        self.bn4 = nn.BatchNorm2d(out_channels)
        self.relu4 = nn.ReLU(inplace=True)
        
        self.se = SEBlock(in_channels, se_ratio)
        
        if self.apply_shortcut and self.stride == 2:
            self.shortcut = nn.Sequential(
                nn.AvgPool2d(2, 2),
                nn.Conv2d(self.in_channels, self.out_channels, 1, bias=False),
                nn.BatchNorm2d(self.out_channels),
            ) 
        
        elif self.apply_shortcut:
            self.shortcut = nn.Sequential(
                nn.Conv2d(self.in_channels, self.out_channels, 1, bias=False),
                nn.BatchNorm2d(self.out_channels),
            ) 
        elif self.stride == 2:
            self.shortcut = nn.Sequential(
                nn.AvgPool2d(2, 2),
                nn.Conv2d(self.in_channels, self.out_channels, 1, bias=False),
                nn.BatchNorm2d(self.out_channels),
            ) 
        else: 
            self.shortcut = nn.Identity() 
              
        
        
    def forward(self, x):
        residual = x 
        residual = self.shortcut(residual)
        x = self.relu1(self.bn1(self.conv1(x)))
        # insert same padding
        filter_size = self.conv2.kernel_size[0]
        pads = get_same_pads(x, filter_size, self.stride)
        x1 = F.pad(x[:, :self.group, :, :], pads)
        x2 = F.pad(x[:, self.group:, :, :], pads)
        x1 = self.relu2(self.bn2(self.conv2(x1)))
        x2 = self.relu3(self.bn3(self.conv2(x2)))
        x = torch.cat((x1, x2), dim=1)
        x = self.se(x)
        x = self.bn4(self.conv4(x))
        x += residual 
        return self.relu4(x)
    
    @property 
    def apply_shortcut(self):
        return self.in_channels != self.out_channels
    
    
    
def get_same_pads(x, filter_size: int, stride):
    _, _, in_height, in_width = x.shape 
    if (in_height % stride == 0):
        pad_along_height = max(filter_size - stride, 0)
    else:
        pad_along_height = max(filter_size - (in_height % stride), 0)
    if (in_width % stride == 0):
        pad_along_width = max(filter_size - stride, 0)
    else:
        pad_along_width = max(filter_size - (in_width % stride), 0)

    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left
    
    return pad_left, pad_right, pad_top, pad_bottom


class SEBlock(nn.Module):
    def __init__(self, in_channels, se_ratio=0.25):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, int(in_channels * se_ratio), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(int(in_channels * se_ratio), in_channels, bias=False),
            nn.Sigmoid(),
        )
        
    def forward(self, x):
        bs, ch, _, _ = x.shape 
        out = self.squeeze(x).view(bs, ch)
        out = self.excitation(out).view(bs, ch, 1, 1)
        out = x * out.expand_as(x)
        return out 
    
    
    
    
    
def main():
    ten = torch.randn(1, 32, 224, 224)
    block = DBlock(32, 64, 1, 4, 1, 0.25)
    out = block(ten)
    print(out.size())
    
    ten = torch.randn(1, 32, 224, 224)
    block = DBlock(32, 64, 1, 4, 2, 0.25)
    out = block(ten)
    print(out.size())
    
    encoder = Encoder()
    ten = torch.randn(1, 3, 224, 224)
    out = encoder(ten)
    print(out.size())
    
    ten = torch.randn(1, 3, 512, 512)
    out = encoder(ten)
    print(out.size())
    
    

if __name__ == "__main__":
    main()