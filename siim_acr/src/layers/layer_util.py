import torch
from torch.autograd import Variable
import torch.nn.functional as F
from layers.loss_funcs.loss import *

class CBAM_Module(nn.Module):
    def __init__(self, channels, reduction,attention_kernel_size=3):
        super(CBAM_Module, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid_channel = nn.Sigmoid()
        k=2
        self.conv_after_concat = nn.Conv2d(k, 1,
                                           kernel_size = attention_kernel_size,
                                           stride=1,
                                           padding = attention_kernel_size//2)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        # Channel attention module
        module_input = x
        avg = self.avg_pool(x)
        mx = self.max_pool(x)
        avg = self.fc1(avg)
        mx = self.fc1(mx)
        avg = self.relu(avg)
        mx = self.relu(mx)
        avg = self.fc2(avg)
        mx = self.fc2(mx)
        x = avg + mx
        x = self.sigmoid_channel(x)
        # Spatial attention module
        x = module_input * x
        module_input = x
        b, c, h, w = x.size()
        avg = torch.mean(x, 1, True)
        mx, _ = torch.max(x, 1, True)
        x = torch.cat((avg, mx), 1)
        x = self.conv_after_concat(x)
        x = self.sigmoid_spatial(x)
        x = module_input * x
        return x
class ConvBn2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3), stride=(1,1), padding=(1,1)):
        super(ConvBn2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
    def forward(self, z):
        x = self.conv(z)
        x = self.bn(x)
        return x

class ConvBnRelu2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3), stride=(1,1), padding=(1,1)):
        super(ConvBnRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
    def forward(self, z):
        x = self.conv(z)
        x = self.bn(x)
        x = F.relu(x, inplace=True)
        return x

class Decoder(nn.Module):
    def __init__(self, in_channels, channels, out_channels ,
                 up_sample=True,
                 attention_type=None,
                 attention_kernel_size=3,
                 reduction=16,
                 reslink=False):
        super(Decoder, self).__init__()
        self.conv1 = ConvBn2d(in_channels,  channels, kernel_size=3, padding=1)
        self.conv2 = ConvBn2d(channels, out_channels, kernel_size=3, padding=1)
        self.up_sample = up_sample
        self.attention = attention_type is not None
        self.attention_type = attention_type
        self.reslink = reslink
        if attention_type is None:
            pass
        elif attention_type.find('cbam') >= 0:
            self.channel_gate = CBAM_Module(out_channels, reduction,attention_kernel_size)
        if self.reslink:
            self.shortcut = ConvBn2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
    def forward(self, x, size=None):
        if self.up_sample:
            if size is None:
                x = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)  # False
            else:
                x = F.upsample(x, size=size, mode='bilinear', align_corners=True)
        if self.reslink:
            shortcut = self.shortcut(x)
        x = F.relu(self.conv1(x), inplace=True)
        x = F.relu(self.conv2(x), inplace=True)
        if self.attention:
            x = self.channel_gate(x)
        if self.reslink:
            x = F.relu(x+shortcut)
        return x
