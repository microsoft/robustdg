import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Variable
from torchvision.models.resnet import ResNet, BasicBlock


# Defining the network (LeNet-5)  
class LeNet5(torch.nn.Module):          
     
    def __init__(self):     
        super(LeNet5, self).__init__()
                
        self.predict_conv_net= nn.Sequential(
                    # Convolution (In LeNet-5, 32x32 images are given as input. Hence padding of 2 is done below)
                    nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0, bias=True),
                    nn.ReLU(),
                    # Max-pooling
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    # Convolution
                    nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0, bias=True),
                    nn.ReLU(),
                    # Max-pooling
                    nn.MaxPool2d(kernel_size=2, stride=2), 
                )
        self.predict_fc_net= nn.Sequential(                    
                    # Fully connected layer
                    # convert matrix with 16*5*5 (= 400) features to a matrix of 120 features (columns)            
                    nn.Linear(16*5*5, 120),   
                    nn.ReLU(),
                    # convert matrix with 120 features to a matrix of 84 features (columns)            
                    nn.Linear(120, 84),       
                    nn.ReLU(),
                    # convert matrix with 84 features to a matrix of 10 features (columns)            
                    nn.Linear(84, 10),
                )
        
    def forward(self, x):        
#         x= x.view(-1, 1, 28, 28)
        out= self.predict_conv_net(x)
        out= out.view(-1,out.shape[1]*out.shape[2]*out.shape[3])
        out= self.predict_fc_net(out)
        return out
    
class ClfNet(nn.Module):
    def __init__(self, rep_net, rep_dim, out_dim):
        super(ClfNet, self).__init__()
        self.rep_net= rep_net
        self.erm_net=nn.Sequential(
                      nn.Linear(rep_dim, out_dim)
#                     nn.Linear(rep_dim, 200),
#                     nn.BatchNorm1d(200),
#                     nn.Dropout(),
#                     nn.ReLU(),
#                     nn.Linear(200, 100),
#                     nn.BatchNorm1d(100),
#                     nn.Dropout(),
#                     nn.ReLU(),
#                     nn.Linear(100, out_dim)
                )
        
    def forward(self, x):
        out= self.rep_net(x)
        return self.erm_net(out)        
