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
			#if isinstance(layer, torch.nn.Conv2d):
				#init.kaiming_normal_(layer.weight, a=0, mode='fan_out')
				#layer.bias.data.zero_()
            if isinstance(layer, torch.nn.Linear):
                init.xavier_uniform_(layer.weight, 0.1)
                layer.bias.data.zero_()	
			#elif isinstance(layer, torch.nn.BatchNorm2d) or isinstance(layer, torch.nn.BatchNorm1d):
				#layer.weight.data.fill_(1)
				#layer.bias.data.zero_()	
    
    def forward(self, x):
        x = self.features(x*57.6)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


def alexnet(classes, pretrained=False, erm_base=1):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AlexNet(classes, erm_base)
    
    if pretrained:
        state_dict = torch.load("models/alexnet_caffe.pth.tar")
        del state_dict["classifier.fc8.weight"]
        del state_dict["classifier.fc8.bias"]
        model.load_state_dict(state_dict, strict = False)
        
    module=[]
    if erm_base==0:    
        for idx in range(4):
            layer= model.classifier[idx]
            module.append(layer)
        model.classifier= nn.Sequential( *module )
    print(model)
    return model


class ClfNet(nn.Module):
    def __init__(self, rep_net, rep_dim, out_dim):
        super(ClfNet, self).__init__()
        self.rep_net= rep_net
        self.erm_net=nn.Sequential(
                    nn.Linear(rep_dim, out_dim),
                    nn.ReLU(),
                    nn.Linear(out_dim, out_dim),
                )
        
    def forward(self, x):
        out= self.rep_net(x)
        return self.erm_net(out)
