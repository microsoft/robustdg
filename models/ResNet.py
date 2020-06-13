from torch import nn
from torch.utils import model_zoo
import torchvision
from torchvision.models.resnet import BasicBlock, model_urls, Bottleneck

class ResNet(nn.Module):
    def __init__(self, block, layers, jigsaw_classes=1000, classes=100, domains=3):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.jigsaw_classifier = nn.Linear(512 * block.expansion, jigsaw_classes)
        self.class_classifier = nn.Linear(512 * block.expansion, classes)
        #self.domain_classifier = nn.Linear(512 * block.expansion, domains)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def is_patch_based(self):
        return False

    def forward(self, x, **kwargs):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.jigsaw_classifier(x), self.class_classifier(x)


def resnet18(pretrained=True, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
    return model

def resnet50(pretrained=True, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
    return model


# bypass layer
class Identity(nn.Module):
    def __init__(self,n_inputs):
        super(Identity, self).__init__()
        self.in_features=n_inputs
        
    def forward(self, x):
        return x

    
def get_resnet(model_name, classes, erm_base, num_ch, pre_trained):
    if model_name == 'resnet18':
        model=  torchvision.models.resnet18(pre_trained)
        n_inputs = model.fc.in_features
        n_outputs= classes
        #print(n_inputs)
#         model.fc = nn.Linear(n_inputs, n_outputs)
        if erm_base:
            model.fc = nn.Linear(n_inputs, n_outputs)
        else:
            model.fc = Identity(n_inputs)
#             model.fc= nn.Sequential( nn.Linear(n_inputs, n_inputs),
#                                      nn.ReLU(),
#                                    )
        
        if num_ch==1:
            model.conv1 = nn.Conv2d(1, 64, 
                                kernel_size=(7, 7), 
                                stride=(2, 2), 
                                padding=(3, 3), 
                            bias=False)
            
    return model

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
