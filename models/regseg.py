import torch
import torch.nn as nn
import torch.nn.functional as F


class RegSeg(nn.Module):
    def __init__(self, num_classes=19):
        super(RegSeg, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(num_classes)

    def forward(self, x):
        x_1_4, x_1_8, x_1_16 = self.encoder(x)
        x = self.decoder(x_1_4, x_1_8, x_1_16)
        return x


class Encoder(nn.Module):
    def __init__(self, out_channels=320):
        super(Encoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.d_block_1_4 = DBlock(32, 48, 1, 1, 2)
        self.d_block_1_8 = nn.Sequential(
            DBlock(48, 48, 1, 1, 2),
            DBlock(48, 48, 1, 1, 1),
            DBlock(48, 128, 1, 1, 1),
        )
        self.d_block_1_16 = nn.Sequential(
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

        x_1_4 = self.d_block_1_4(x)
        x_1_8 = self.d_block_1_8(x_1_4)
        x_1_16 = self.d_block_1_16(x_1_8)

        return x_1_4, x_1_8, x_1_16


class Decoder(nn.Module):
    def __init__(self, num_classes):
        super(Decoder, self).__init__()
        self.conv1_4 = nn.Sequential(
            nn.Conv2d(48, 8, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )
        self.conv1_8 = nn.Sequential(
            nn.Conv2d(128, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.conv1_16 = nn.Sequential(
            nn.Conv2d(320, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(72, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(64, num_classes, 1),
            nn.BatchNorm2d(num_classes),
        )
        self.upsample_1_16 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.upsample_1_8 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.final_upsample = nn.UpsamplingBilinear2d(scale_factor=4)


    def forward(self, x_1_4, x_1_8, x_1_16):
        x_1_4 = self.conv1_4(x_1_4)
        x_1_8 = self.conv1_8(x_1_8)
        x_1_16 = self.conv1_16(x_1_16)
        x_1_16 = self.upsample_1_16(x_1_16)
        x_1_8 = x_1_8 + x_1_16

        filter_size = self.conv3[0].kernel_size[0]
        pads = get_same_pads(x_1_8, filter_size, 1)
        x_1_8 = F.pad(x_1_8, pads)
        x_1_8 = self.conv3(x_1_8)

        x_1_8 = self.upsample_1_8(x_1_8)
        x = torch.cat((x_1_4, x_1_8), dim=1)

        filter_size = self.conv4[0].kernel_size[0]
        pads = get_same_pads(x, filter_size, 1)
        x = F.pad(x, pads)
        x = self.conv4(x)
        x = self.conv5(x)

        return self.final_upsample(x)



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

