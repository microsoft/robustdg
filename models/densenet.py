from torch import nn
from torch.utils import model_zoo
import torchvision
from torchvision.models.resnet import BasicBlock, model_urls, Bottleneck

import torchxrayvision as xrv

# bypass layer
class Identity(nn.Module):
    def __init__(self,n_inputs):
        super(Identity, self).__init__()
        self.in_features=n_inputs
        
    def forward(self, x):
        return x

    
def get_densenet(model_name, classes, fc_layer, num_ch, pre_trained, os_env):
    if model_name == 'densenet121':
        model= xrv.models.DenseNet(num_classes=classes, in_channels=num_ch, 
                                        **xrv.models.get_densenet_params('densenet121'))         
        n_inputs = model.classifier.in_features
        n_outputs= classes
        
    if fc_layer:
        model.classifier = nn.Linear(n_inputs, n_outputs)
    else:
        print('DenseNet Contrastive')
        model.classifier = Identity(n_inputs)
#         model.fc= nn.Sequential( nn.Linear(n_inputs, n_inputs),
#                                  nn.ReLU(),
#                                )
        
    return model