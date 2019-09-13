from layers.scheduler import *
from layers.backbone.senet import *
from utils.common_util import *
from layers.layer_util import *

## net  ######################################################################
class SEnetUnet(nn.Module):
    def load_pretrain(self, pretrain_file):
        print('load pretrained file: %s' % pretrain_file)
        self.backbone.load_state_dict(torch.load(pretrain_file, map_location=lambda storage, loc: storage))
    #attention_type
    #none:no attention
    #scse_v0:1803.02579 Concurrent Spatial and Channel  Squeeze & Excitation in Fully Convolutional Networks.pdf
    #https://github.com/Youngkl0726/Convolutional-Block-Attention-Module/blob/master/CBAMNet.py
    #https://github.com/kobiso/CBAM-keras/blob/master/models/attention_module.py
    def __init__(self,
                 feature_net='se_resnext50_32x4d',
                 attention_type=None,
                 position_encode=False,
                 reduction=16,
                 pretrained_file=None,
                 ):
        super().__init__()
        self.attention = attention_type is not None
        self.attention_type = attention_type
        self.position_encode = position_encode

        if attention_type == 'cbam_v0a':
            decoder_kernels = [1, 1, 1, 1, 1]
        else:
            decoder_kernels = [1, 1, 1, 1, 1]
        if feature_net == 'se_resnext50_32x4d':
            self.backbone = se_resnext50_32x4d()
            self.EX = 4
        if feature_net == 'se_resnext101_32x4d':
            self.backbone = se_resnext101_32x4d()
            self.EX = 4
        elif feature_net == 'se_resnet50':
            self.backbone = se_resnet50()
            self.EX = 4
        elif feature_net == 'se_resnet101':
            self.backbone = se_resnet101()
            self.EX = 4
        elif feature_net == 'se_resnet152':
            self.backbone = se_resnet152()
            self.EX = 4
        elif feature_net == 'senet154':
            self.backbone = senet154()
            self.EX = 4

        self.load_pretrain(pretrained_file)
        self.conv1 =nn.Sequential(*list(self.backbone.layer0.children())[:-1])
        self.encoder2 = self.backbone.layer1  # 64*self.EX
        self.encoder3 = self.backbone.layer2  # 128*self.EX
        self.encoder4 = self.backbone.layer3  # 256*self.EX
        self.encoder5 = self.backbone.layer4  # 512*self.EX
        self.center = nn.Sequential(
            ConvBn2d(512*self.EX, 512*self.EX, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ConvBn2d(512*self.EX, 256*self.EX, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        att_type=self.attention_type
        self.decoder5 = Decoder(512*self.EX + 256*self.EX, 512, 32,
                                attention_type=att_type,
                                attention_kernel_size=decoder_kernels[0],
                                position_encode=position_encode,
                                reduction=reduction)
        self.decoder4 = Decoder(256*self.EX + 32, 256, 32,
                                attention_type=att_type,
                                attention_kernel_size=decoder_kernels[1],
                                position_encode=position_encode,
                                reduction=reduction)
        self.decoder3 = Decoder(128*self.EX + 32, 128, 32,
                                attention_type=att_type,
                                attention_kernel_size=decoder_kernels[2],
                                position_encode=position_encode,
                                reduction=reduction)
        self.decoder2 = Decoder(64*self.EX + 32, 64, 32,
                                attention_type=att_type,
                                attention_kernel_size=decoder_kernels[3],
                                position_encode=position_encode,
                                reduction=reduction)
        self.decoder1 = Decoder(32, 32, 32,
                                attention_type=att_type,
                                attention_kernel_size=decoder_kernels[4],
                                position_encode=position_encode,
                                reduction=reduction)
        self.logit = nn.Sequential(
            ConvBnRelu2d(160, 64, kernel_size=3, padding=1),
            ConvBnRelu2d(64, 32, kernel_size=3, padding=1),
            nn.Conv2d(32, 1, kernel_size=1, padding=0),
        )
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

def unet_se_resnext50_cbam_v0a(**kwargs):
    pretrained_file = kwargs['pretrained_file']
    model = SEnetUnet(feature_net='se_resnext50_32x4d', attention_type='cbam_v0a', pretrained_file=pretrained_file)
    return model
