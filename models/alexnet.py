#Module taken from the repository G2DM repository for AlexNet architecture specific to PACS: https://github.com/belaalb/G2DM
import os

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Variable


import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.init as init

from collections import OrderedDict

__all__ = ['AlexNet', 'alexnet']

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class Id(nn.Module):
    def __init__(self):
        super(Id, self).__init__()

    def forward(self, x):
        return x


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, dropout=True):
        super(AlexNet, self).__init__()
        
        self.features = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(3, 96, kernel_size=11, stride=4)),
            ("relu1", nn.ReLU(inplace=True)),
            ("pool1", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
            ("norm1", nn.LocalResponseNorm(5, 1.e-4, 0.75)),
            ("conv2", nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2)),
            ("relu2", nn.ReLU(inplace=True)),
            ("pool2", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
            ("norm2", nn.LocalResponseNorm(5, 1.e-4, 0.75)),
            ("conv3", nn.Conv2d(256, 384, kernel_size=3, padding=1)),
            ("relu3", nn.ReLU(inplace=True)),
            ("conv4", nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2)),
            ("relu4", nn.ReLU(inplace=True)),
            ("conv5", nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2)),
            ("relu5", nn.ReLU(inplace=True)),
            ("pool5", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
        ]))
        
        self.classifier = nn.Sequential(OrderedDict([
                ("fc6", nn.Linear(256 * 6 * 6, 4096)),
                ("relu6", nn.ReLU(inplace=True)),
                ("drop6", nn.Dropout()),
                ("fc7", nn.Linear(4096, 4096)),
                ("relu7", nn.ReLU(inplace=True)),
                ("drop7", nn.Dropout()),
                ("fc8", nn.Linear(4096, num_classes))
        ]))
        
        self.initialize_params()        
        

    def initialize_params(self):

        for layer in self.modules():
            if isinstance(layer, torch.nn.Linear):
                init.xavier_uniform_(layer.weight, 0.1)
                layer.bias.data.zero_()	
    
    def forward(self, x):
        x = self.features(x*57.6)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


def alexnet(model_name, classes, fc_layer, num_ch, pre_trained, os_env):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AlexNet(classes)
    
    if pre_trained:
        if os_env:
            state_dict = torch.load( os.getenv('PT_DATA_DIR') + "/pacs/alexnet_caffe.pth.tar")
        else:
            state_dict = torch.load("/home/t-dimaha/RobustDG/robustdg/data/datasets/pacs/alexnet_caffe.pth.tar")

        del state_dict["classifier.fc8.weight"]
        del state_dict["classifier.fc8.bias"]
        model.load_state_dict(state_dict, strict = False)
        
    return model