import torch
from torch import nn
from torch.utils import model_zoo
import torchvision
from torchvision.models.resnet import BasicBlock, model_urls, Bottleneck
import os

class GroupNorm(torch.nn.GroupNorm):
    def __init__(self, num_channels, num_groups=32, **kwargs):
        super().__init__(num_groups, num_channels, **kwargs)

# bypass layer
class Identity(nn.Module):
    def __init__(self,n_inputs):
        super(Identity, self).__init__()
        self.in_features=n_inputs
        
    def forward(self, x):
        return x

    
def get_resnet(model_name, classes, fc_layer, num_ch, pre_trained, dp_noise, os_env):    
    if model_name == 'resnet18':
        if os_env:        
            model=  torchvision.models.resnet18()
            if pre_trained:
                model.load_state_dict(torch.load( os.getenv('PT_DATA_DIR') + '/checkpoints/resnet18-5c106cde.pth' ))
        else:
            if dp_noise:
                #TODO
                #This cannot work with pre trained model, as the batch norm weights have been replaced by group norm
                model=  torchvision.models.resnet18(pre_trained, norm_layer=GroupNorm)
#                 model=  torchvision.models.resnet18(pre_trained)
            else:
                model=  torchvision.models.resnet18(pre_trained)
            
        n_inputs = model.fc.in_features
        n_outputs= classes
        
    elif model_name == 'resnet50':
        if os_env:        
            model=  torchvision.models.resnet50()
            if pre_trained:
                model.load_state_dict(torch.load( os.getenv('PT_DATA_DIR') + '/checkpoints/resnet50-19c8e357.pth' ))
        else:
            if dp_noise:
                model=  torchvision.models.resnet50(pre_trained, norm_layer=GroupNorm)                
            else:
                model=  torchvision.models.resnet50(pre_trained)
            
        n_inputs = model.fc.in_features
        n_outputs= classes
        
    if fc_layer:
        model.fc = nn.Linear(n_inputs, n_outputs)
    else:
        print('Here')
        model.fc = Identity(n_inputs)
#         model.fc= nn.Sequential( nn.Linear(n_inputs, n_inputs),
#                                  nn.ReLU(),
#                                )
        
    if num_ch==1:
        model.conv1 = nn.Conv2d(1, 64, 
                                kernel_size=(7, 7), 
                                stride=(2, 2), 
                                padding=(3, 3), 
                        bias=False)
    return model