from config.config import *
from utils.common_util import *
from networks.unet_resnet import *
from networks.unet_senet import *

model_names = {
    'unet_resnet34_cbam_v0a': 'resnet34-333f7ec4.pth',
    'unet_se_resnext50_cbam_v0a': 'se_resnext50_32x4d-a260b3a4.pth',
}

def init_network(params):
    architecture = params.get('architecture', 'unet_resnet34_cbam_v0a')
    pretrained_file = opj(PRETRAINED_DIR, model_names[architecture])
    print(">> Using pre-trained model.")
    net = eval(architecture)(pretrained_file=pretrained_file)
    return net
