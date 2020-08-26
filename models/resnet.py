from torch import nn
from torch.utils import model_zoo
import torchvision
from torchvision.models.resnet import BasicBlock, model_urls, Bottleneck

# bypass layer
class Identity(nn.Module):
    def __init__(self,n_inputs):
        super(Identity, self).__init__()
        self.in_features=n_inputs
        
    def forward(self, x):
        return x

    
def get_resnet(model_name, classes, fc_layer, num_ch, pre_trained):
    if model_name == 'resnet18':
        model=  torchvision.models.resnet18(pre_trained)
        n_inputs = model.fc.in_features
        n_outputs= classes
    elif model_name == 'resnet50':
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