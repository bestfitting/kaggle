import torch.nn.functional as F
from layers.scheduler import *
from layers.backbone.resnet import *
from utils.common_util import *
from layers.layer_util import *

## net  ######################################################################
class ResnetUnet(nn.Module):

    def load_pretrain(self, pretrain_file):
        print('load pretrained file: %s' % pretrain_file)
        self.resnet.load_state_dict(torch.load(pretrain_file, map_location=lambda storage, loc: storage))

    def __init__(self,feature_net='resnet34',
                 attention_type=None,
                 reduction=16,
                 reslink=False,
                 out_channel=1,
                 pretrained_file=None,
                 ):
        super().__init__()
        self.attention = attention_type is not None
        self.attention_type = attention_type
        self.out_channel = out_channel
        decoder_kernels = [1, 1, 1, 1, 1]
        if feature_net == 'resnet18':
            self.resnet = resnet18()
            self.EX = 1
        elif feature_net == 'resnet34':
            self.resnet = resnet34()
            self.EX = 1
        elif feature_net == 'resnet50':
            self.resnet = resnet50()
            self.EX = 4
        elif feature_net == 'resnet101':
            self.resnet = resnet101()
            self.EX = 4
        elif feature_net == 'resnet152':
            self.resnet = resnet152()
            self.EX = 4

        self.load_pretrain(pretrained_file)
        self.conv1 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
        )
        self.encoder2 = self.resnet.layer1
        self.encoder3 = self.resnet.layer2
        self.encoder4 = self.resnet.layer3
        self.encoder5 = self.resnet.layer4
        att_type=self.attention_type
        self.decoder4 = Decoder(256*self.EX + 32, 256, 32,
                                attention_type=att_type,
                                attention_kernel_size=decoder_kernels[1],
                                reduction=reduction,
                                reslink=reslink)
        self.decoder3 = Decoder(128*self.EX + 32, 128, 32,
                                attention_type=att_type,
                                attention_kernel_size=decoder_kernels[2],
                                reduction=reduction,
                                reslink=reslink)
        self.decoder2 = Decoder(64*self.EX + 32, 64, 32,
                                attention_type=att_type,
                                attention_kernel_size=decoder_kernels[3],
                                reduction=reduction,
                                reslink=reslink)
        self.decoder1 = Decoder(32, 32, 32,
                                attention_type=att_type,
                                attention_kernel_size=decoder_kernels[4],
                                reduction=reduction,
                                reslink=reslink)

        self.logit = nn.Sequential(
            ConvBnRelu2d(160, 64, kernel_size=3, padding=1),
            ConvBnRelu2d(64, 32, kernel_size=3, padding=1),
            nn.Conv2d(32, out_channel, kernel_size=1, padding=0),
        )

        center_channels = 512 * self.EX
        decoder5_channels = 512 * self.EX + 256 * self.EX

        self.center = nn.Sequential(
            ConvBn2d(center_channels, 512 * self.EX, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ConvBn2d(512 * self.EX, 256 * self.EX, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder5 = Decoder(decoder5_channels, 512, 32,
                                attention_type=att_type,
                                attention_kernel_size=decoder_kernels[0],
                                reduction=reduction,
                                reslink=reslink)

    def forward(self, x, *args):
        x = self.conv1(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        e2 = self.encoder2(x)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)

        f = self.center(e5)

        d5 = self.decoder5(torch.cat([f, e5], 1))
        d4 = self.decoder4(torch.cat([d5, e4], 1))
        d3 = self.decoder3(torch.cat([d4, e3], 1))
        d2 = self.decoder2(torch.cat([d3, e2], 1))
        d1 = self.decoder1(d2)
        f = torch.cat((
                 d1,
                 F.upsample(d2, scale_factor=2, mode='bilinear',align_corners=False),
                 F.upsample(d3, scale_factor=4, mode='bilinear', align_corners=False),
                 F.upsample(d4, scale_factor=8, mode='bilinear', align_corners=False),
                 F.upsample(d5, scale_factor=16, mode='bilinear', align_corners=False),
        ), 1)
        logit = self.logit(f)
        return logit

def unet_resnet34_cbam_v0a(**kwargs):
    pretrained_file = kwargs['pretrained_file']
    model = ResnetUnet(feature_net='resnet34', attention_type='cbam_v0a', pretrained_file=pretrained_file)
    return model
