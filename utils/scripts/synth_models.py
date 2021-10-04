import sys, copy
import torch, torchvision
from torch import optim, nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import utils.scripts.gendata as gendata
import utils.scripts.utils as utils
import numpy as np
import utils.scripts.gpu_utils as gu
import utils.scripts.ptb_utils as pu 

def kaiming_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        nn.init.kaiming_uniform_(m.bias.data)

class SequenceClassifier(nn.Module):

    def __init__(self, seq_model, idim, hdim, hl, input_size, num_classes=2, many_to_many=False, unsqueeze_input=True):
        super(SequenceClassifier, self).__init__()
        self.seq_model = seq_model
        self.hdim = hdim
        self.hl = hl
        self.input_size = input_size
        self.idim = idim
        self.num_classes = num_classes
        self.unsqueeze_input = unsqueeze_input
        self.many_to_many = many_to_many

        self.seq_length = self.idim//self.input_size
        self.seq = self.seq_model(input_size=input_size, hidden_size=hdim, num_layers=hl, batch_first=True)
        self.lin_idim = hdim*self.seq_length if many_to_many else hdim
        self.lin = nn.Linear(self.lin_idim, num_classes)

    def forward(self, x):
        if self.unsqueeze_input: x = x.unsqueeze(2)
        bsize, idim, _ = x.shape
        seq_length = idim//self.input_size
        x = x.view((bsize, seq_length, self.input_size))
        out, hidden = self.seq(x)
        lin_in = out[:,-1,:]
        if self.many_to_many: lin_in = out.contiguous().view((bsize, -1))
        lin_out = self.lin(lin_in)
        return lin_out

class GRUClassifier(SequenceClassifier):

    def __init__(self, idim, hdim, hl, input_size, num_classes=2, many_to_many=False, unsqueeze_input=True):
        super(GRUClassifier, self).__init__(nn.GRU, idim, hdim, hl, input_size, many_to_many=many_to_many, num_classes=num_classes, unsqueeze_input=unsqueeze_input)

class LSTMClassifier(SequenceClassifier):

    def __init__(self, idim, hdim, hl, input_size, num_classes=2, many_to_many=False, unsqueeze_input=True):
        super(LSTMClassifier, self).__init__(nn.LSTM, idim, hdim, hl, input_size, many_to_many=many_to_many, num_classes=num_classes, unsqueeze_input=unsqueeze_input)

class CNNClassifier(nn.Module):

    def __init__(self, out_channels, hl, kernel_size, idim, num_classes=2, padding=None, stride=1, maxpool_kernel_size=None, use_maxpool=False):
        """
        Fixed architecture:
        - default max pool kernel size half of convolution kernel size
        - default padding = kernel size - 1 // 2 to maintain same dimension
        - stride = 1
        - 1 FC layer
        """
        if padding == None: assert kernel_size % 2 == 1, "use odd kernel size, equal padding constraint"
        super(CNNClassifier, self).__init__()
        self.out_channels = out_channels
        self.num_conv = hl
        self.kernel_size = kernel_size
        self.padding = padding or (self.kernel_size-1)//2
        self.stride = 1
        self.num_classes = 2
        self.idim = idim
        self.use_maxpool = use_maxpool
        self.maxpool_kernel_size = maxpool_kernel_size or self.kernel_size//2

        self.maxpool = nn.MaxPool1d(self.maxpool_kernel_size)
        self.ih_conv = nn.Conv1d(1, self.out_channels, self.kernel_size, padding=self.padding, stride=self.stride)

        self.hh_convs = []
        for _ in range(self.num_conv-1):
            self.hh_convs.append(nn.Conv1d(self.out_channels, self.out_channels, self.kernel_size, padding=self.padding, stride=self.stride))
            self.hh_convs.append(nn.ReLU())
        self.hh_convs = nn.Sequential(*self.hh_convs)

        fc_idim = int(self.idim/self.maxpool_kernel_size) if self.use_maxpool else self.idim
        self.fc_layer = nn.Linear(self.out_channels*fc_idim, self.idim)
        self.out_layer = nn.Linear(self.idim, self.num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        bs = x.shape[0]
        x_ = x.unsqueeze(1)

        x_ = self.relu(self.ih_conv(x_))
        x_ = self.hh_convs(x_)

        if self.use_maxpool:  x_ = self.maxpool(x_)
        x_ = self.relu(self.fc_layer(x_.view(bs, -1)))

        return self.out_layer(x_)

class CNN2DClassifier(nn.Module):

    def __init__(self, num_filters, filter_size, num_layers, input_shape, input_channels=1, stride=2, padding=None, num_stride2_layers=2, fc_idim=None, fc_odim=None, num_classes=2, use_avgpool=True, avgpool_ksize=5):
        super(CNN2DClassifier, self).__init__()
        self.outch = num_filters
        self.fsize = filter_size
        self.input_channels = input_channels
        self.hl = num_layers
        self.padding = (self.fsize-1)//2 if padding is None else padding
        self.num_classes = num_classes
        num_stride2_layers = num_stride2_layers
        self.strides = iter([stride]*num_stride2_layers+[1]*(num_layers-num_stride2_layers))
        self.use_avgpool = use_avgpool
        self.avgpool_ksize = avgpool_ksize

        self.convs = [nn.Conv2d(self.input_channels, self.outch, self.fsize, padding=self.padding, stride=next(self.strides)), nn.ReLU()]
        if self.use_avgpool: self.convs.append(nn.AvgPool2d(self.avgpool_ksize))

        for _ in range(self.hl-1):
            self.convs.append(nn.Conv2d(self.outch, self.outch, self.fsize, stride=next(self.strides), padding=self.padding))
            self.convs.append(nn.ReLU())
            if self.use_avgpool: self.convs.append(nn.AvgPool2d(self.avgpool_ksize))

        self.convs = nn.Sequential(*self.convs) # need to wrap for gpu
        sl = min(self.hl, num_stride2_layers)
        self.fc_idim = int(num_filters*input_shape[0]*input_shape[1]/float(4**sl)) if fc_idim is None else fc_idim
        self.fc_odim = fc_odim if fc_odim is not None else self.fc_idim
        self.fc = nn.Linear(self.fc_idim, self.fc_odim)
        self.out = nn.Linear(self.fc_odim, self.num_classes)

    def forward(self, x):
        x = self.convs(x)
        x = x.reshape(x.shape[0], -1)
        return self.out(F.relu(self.fc(x)))

def get_linear(input_dim, num_classes):
    return nn.Sequential(nn.Linear(input_dim, num_classes))

def get_fcn(idim, hdim, odim, hl=1, init=False, activation=nn.ReLU, use_activation=True, use_bn=False, input_dropout=0, dropout=0):
    use_dropout = dropout > 0
    layers = []
    if input_dropout > 0: layers.append(nn.Dropout(input_dropout))
    layers.append(nn.Linear(idim, hdim))
    if use_activation: layers.append(activation())
    if use_dropout: layers.append(nn.Dropout(dropout))
    if use_bn: layers.append(nn.BatchNorm1d(hdim))
    for _ in range(hl-1):
        l = [nn.Linear(hdim, hdim)]
        if use_activation: l.append(activation())
        if use_dropout: l.append(nn.Dropout(dropout))
        if use_bn: l.append(nn.BatchNorm1d(hdim))
        layers.extend(l)
    layers.append(nn.Linear(hdim, odim))
    model = nn.Sequential(*layers)

    if init: model.apply(kaiming_init)
    return model