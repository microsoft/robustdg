import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Variable
from torchvision.models.resnet import ResNet, BasicBlock

# Defining the network (IRM MNIST Model)
class IrmMnist(torch.nn.Module):
     
    def __init__(self):     
        super(IrmMnist, self).__init__()
        
        if flags.grayscale_model:
            lin1 = nn.Linear(14 * 14, flags.hidden_dim)
        else:
            lin1 = nn.Linear(2 * 14 * 14, flags.hidden_dim)
            
        lin2 = nn.Linear(flags.hidden_dim, flags.hidden_dim)
        lin3 = nn.Linear(flags.hidden_dim, 1)
        for lin in [lin1, lin2, lin3]:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
            
        self._main = nn.Sequential(lin1, nn.ReLU(True), lin2, nn.ReLU(True), lin3)
        
    def forward(self, input):
        if flags.grayscale_model:
            out = input.view(input.shape[0], 2, 14 * 14).sum(dim=1)
        else:
            out = input.view(input.shape[0], 2 * 14 * 14)
            
        out = self._main(out)
        return out
