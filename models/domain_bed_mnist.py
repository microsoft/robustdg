import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Variable

"""
MNIST CNN architecture from the the paper DomainBed: https://github.com/facebookresearch/DomainBed
"""
class DomainBed(torch.nn.Module):          
     
    def __init__(self, num_ch, fc_layer):     
        super(DomainBed, self).__init__()
        print('DomainBed CNN')
        
        self.n_outputs = 128
        self.num_classes= 10
        self.fc_layer= fc_layer
    
        self.conv1 = nn.Conv2d(num_ch, 64, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, 1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, padding=1)

        self.bn0 = nn.GroupNorm(8, 64)
        self.bn1 = nn.GroupNorm(8, 128)
        self.bn2 = nn.GroupNorm(8, 128)
        self.bn3 = nn.GroupNorm(8, 128)

        self.classifier = nn.Linear(self.n_outputs, self.num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn0(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.bn2(x)

        x = self.conv4(x)
        x = F.relu(x)
        x = self.bn3(x)

        x = x.mean(dim=(2,3))
        
        if self.fc_layer:
            return self.classifier(x)
        else:
            return x