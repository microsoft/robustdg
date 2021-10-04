import os, copy, pickle, time
import random, itertools
from collections import defaultdict, Counter, OrderedDict
import numpy as np
import torch
import pandas as pd
import torchvision
from torch.utils.data import TensorDataset, DataLoader
from torch import optim, nn
import torch.nn.functional as F
import dill
import gpu_utils as gu
import data_utils as du
import synth_models as sm
import utils

class Ensemble(nn.Module):

    def _get_dummy_classifier(self):
        def dummy(x):
            return x
        return dummy

    def __init__(self, models, num_classes, use_softmax=False):
        super(Ensemble, self).__init__()
        self.num_classes = num_classes
        self.use_softmax = use_softmax

        # register models as pytorch modules 
        self.models = []
        for idx, m in enumerate(models,1): 
            setattr(self, 'm{}'.format(idx), m.eval())
            self.models.append(getattr(self, 'm{}'.format(idx)))

        self.classifier = self._get_dummy_classifier()

    def _forward(self, x):
        return x

    def forward(self, x):
        outs = self._forward(x)
        return self.classifier(outs)
       
    def get_output_loader(self, dl, device=gu.get_device(None), bs=None):
        """return dataloader of model output (logit or softmax prob)"""
        X, Y = [], []
        with torch.no_grad():
            for xb, yb in dl:
                xb = xb.to(device)
                out = self._forward(xb).cpu()
                X.append(out)
                Y.append(yb)
        X, Y = torch.cat(X), torch.cat(Y)
        return DataLoader(TensorDataset(X, Y), batch_size=bs or dl.batch_size)

    def fit_classifier(self, tr_dl, te_dl, lr=0.05, adam=False, wd=5e-5, device=None, **fit_kw):
        device = gu.get_device(None) if device is None else device
        self = self.to(device)
        
        c = dict(gap=1000, epsilon=1e-2, wd=5e-5, is_loss_epsilon=True)
        c.update(**fit_kw)

        tro_dl = self.get_output_loader(tr_dl, device)
        teo_dl = self.get_output_loader(te_dl, device)

        if adam: opt = optim.Adam(self.classifier.parameters())
        else: opt = optim.SGD(self.classifier.parameters(), lr=lr, weight_decay=wd)
        stats = utils.fit_model(self.classifier, F.cross_entropy, opt, tro_dl, teo_dl, device=device, **c)

        self.classifier = stats['best_model'][-1].to(device)
        self = self.cpu() 
        return stats

class EnsembleLinear(Ensemble):

    def _get_classifier(self):
        # linear with equal weights and zero bias
        nl = self.num_classes*len(self.models)
        linear = nn.Linear(nl, self.num_classes, bias=self.use_bias)
        nn.init.ones_(linear.weight.data)
        linear.weight.data /= float(nl)
        if self.use_bias: linear.bias.data.zero_()
        return linear

    def __init__(self, models, num_classes=2, use_softmax=False, use_bias=True):
        super(EnsembleLinear, self).__init__(models, num_classes, use_softmax)
        self.use_bias = use_bias
        self.classifier = self._get_classifier()

    def _forward(self, x):
        outs = [m(x) for m in self.models]
        if self.use_softmax: outs = [F.softmax(o, dim=1) for o in outs]
        outs = torch.stack(outs, dim=2)
        outs = outs.reshape(outs.shape[0], -1)
        return outs
    
class EnsembleMLP(Ensemble):

    def _get_classifier(self):
        nl = self.num_classes*len(self.models)
        fcn = sm.get_fcn(nl, self.hdim or nl, self.num_classes, hl=self.hl)
        return fcn

    def __init__(self, models, num_classes=2, use_softmax=False, hdim=None, hl=1):
        super(EnsembleMLP, self).__init__(models, num_classes, use_softmax)
        self.hdim = hdim
        self.hl = hl
        self.classifier = self._get_classifier()

    def _forward(self, x):
        outs = [m(x) for m in self.models]
        if self.use_softmax: outs = [F.softmax(o, dim=1) for o in outs]
        outs = torch.stack(outs, dim=2)
        outs = outs.reshape(outs.shape[0], -1)
        return outs

class EnsembleAverage(Ensemble):

    def __init__(self, models, num_classes=2, use_softmax=False):
        super(EnsembleAverage, self).__init__(models, num_classes, use_softmax)
        self.classifier = self._get_dummy_classifier()

    def _forward(self, x):
        outs = [m(x) for m in self.models]
        if self.use_softmax: outs = [F.softmax(o, dim=1) for o in outs]
        outs = torch.stack(outs)
        return outs.mean(dim=0)

    def fit_classifier(self, *args, **kwargs):
        return None 
