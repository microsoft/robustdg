import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Variable


class SlabClf(nn.Module):
    def __init__(self, inp_shape, out_shape):
		    
        super(SlabClf, self).__init__()
        self.inp_shape = inp_shape
        self.out_shape = out_shape
        self.hidden_dim = 100
        self.feat_net= nn.Sequential(
                     nn.Linear( self.inp_shape, self.hidden_dim),
                     nn.ReLU(),
                    )
        
        self.fc= nn.Sequential(
                    nn.Linear( self.hidden_dim, self.hidden_dim),
                    nn.Linear( self.hidden_dim, self.out_shape),
                    )

        self.disc= nn.Sequential(
                    nn.Linear( self.hidden_dim, self.hidden_dim),
                    nn.Linear( self.hidden_dim, 2),
                    )
        
        self.embedding = nn.Embedding(2, self.hidden_dim)
            
    def forward(self, x):
        return self.fc(self.feat_net(x))