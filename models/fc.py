import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Variable
from torchvision.models.resnet import ResNet, BasicBlock

# Defining the network (LeNet-5)  
class FC(torch.nn.Module):          
     
    def __init__(self, classes, fc_layer):     
        super(FC, self).__init__()
                
        #TODO: Currently Hardcoded for adult dataset
        inp_dim= 29
        self.fc_layer= fc_layer
        self.rep_net= nn.Sequential(                    
                    nn.Linear(inp_dim, inp_dim),
                    nn.ReLU(),
                    nn.Linear(inp_dim, inp_dim)
        )
        
        self.fc_net= nn.Sequential(                    
                    nn.ReLU(),
                    nn.Linear(inp_dim, classes)
        )
        
        
    def forward(self, x):        
        out= self.rep_net(x)
        if self.fc_layer:
            out= self.fc_net(out)
        return out