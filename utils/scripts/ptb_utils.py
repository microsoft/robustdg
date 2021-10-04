import seaborn as sns
import utils
import random
import os, copy, pickle, time
import itertools
from collections import defaultdict, Counter, OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from torch import optim, nn
import torch.nn.functional as F

import utils.scripts.gpu_utils as gu
import utils.scripts.data_utils as du
import utils.scripts.synth_models as synth_models

#import foolbox as fb
#from autoattack import AutoAttack

# Misc
def get_yhat(model, data): return torch.argmax(model(data), 1)
def get_acc(y,yhat): return (y==yhat).sum().item()/float(len(y))

class PGD_Attack(object):

    def __init__(self, eps, lr, num_iter, loss_type, rand_eps=1e-3,
                 num_classes=2, bounds=(0.,1.), minimal=False, restarts=1, device=None):
        self.eps = eps
        self.lr = lr
        self.num_iter = num_iter
        self.B = bounds
        self.restarts = restarts
        self.rand_eps = rand_eps
        self.device = device or gu.get_device(None)
        self.loss_type = loss_type
        self.num_classes = num_classes
        self.classes = list(range(self.num_classes))
        self.delta = None
        self.minimal = minimal # early stop + no eps
        self.project = not self.minimal
        self.loss = -np.inf

    def evaluate_attack(self, dl, model):
        model = model.to(self.device)
        Xa, Ya, Yh, P = [], [], [], []

        for xb, yb in dl:
            xb, yb = xb.to(self.device), yb.to(self.device)
            delta = self.perturb(xb, yb, model)
            xba = xb+delta
            
            with torch.no_grad():
                out = model(xba).detach()
            yh = torch.argmax(out, dim=1)
            xb, yb, yh, xba, delta = xb.cpu(), yb.cpu(), yh.cpu(), xba.cpu(), delta.cpu()
            
            Ya.append(yb)
            Yh.append(yh)
            Xa.append(xba)
            P.append(delta)

        Xa, Ya, Yh, P = map(torch.cat, [Xa, Ya, Yh, P])
        ta_dl = utils._to_dl(Xa, Ya, dl.batch_size)
        acc, loss = utils.compute_loss_and_accuracy_from_dl(ta_dl, model,
                                                            F.cross_entropy,
                                                            device=self.device)
        return {
            'acc': acc.item(),
            'loss': loss.item(),
            'ta_dl': ta_dl,
            'Xa': Xa.numpy(),
            'Ya': Ya.numpy(),
            'Yh': Yh.numpy(),
            'P': P.numpy()
        }

    def perturb(self, xb, yb, model, cpu=False):
        model, xb, yb = model.to(self.device), xb.to(self.device), yb.to(self.device)
        if self.eps == 0: return torch.zeros_like(xb)

        # compute perturbations and track best perturbations
        self.loss = -np.inf
        max_delta = self._perturb_once(xb, yb, model)
        
        with torch.no_grad(): 
            out = model(xb+max_delta)
            max_loss = nn.CrossEntropyLoss(reduction='none')(out, yb)

        for _ in range(self.restarts-1):
            delta = self._perturb_once(xb, yb, model)

            with torch.no_grad():
                out = model(xb+delta)
                all_loss = nn.CrossEntropyLoss(reduction='none')(out, yb)

            loss_flag = all_loss >= max_loss
            max_delta[loss_flag] = delta[loss_flag]
            max_loss = torch.max(max_loss, all_loss)

        if cpu: max_delta = max_delta.cpu()
        return max_delta

    def _perturb_once(self, xb, yb, model, track_scores=False, stop_const=1e-5):
        self.delta = self._init_delta(xb, yb)
        scores = []

        # (minimal) mask perturbations if model already misclassifies
        for t in range(self.num_iter):
            loss, out = self._get_loss(xb, yb, model, get_scores=True)
    
            if self.minimal:
                yh = torch.argmax(out, dim=1).detach()
                not_flipped = yh == yb            
                not_flipped_ratio = not_flipped.sum().item()/float(len(yb))
            else:
                not_flipped = None
                not_flipped_ratio = 1.0

            # stop if almost all examples in the batch misclassified    
            if not_flipped_ratio < stop_const: 
                break
            
            if track_scores: 
                scores.append(out.detach().cpu().numpy())
                
            # compute loss, update + clamp delta
            loss.backward()
            self.loss = max(self.loss, loss.item())

            self.delta = self._update_delta(xb, yb, update_mask=not_flipped)
            self.delta = self._clamp_input(xb, yb)
            
        d = self.delta.detach()

        if track_scores: 
            scores = np.stack(scores).swapaxes(0, 1)
            return d, scores
        
        return d

    def _init_delta(self, xb, yb):
        delta = torch.empty_like(xb)
        delta = delta.uniform_(-self.rand_eps, self.rand_eps)
        delta = delta.to(self.device)
        delta.requires_grad = True
        return delta

    def _clamp_input(self, xb, yb):
        # clamp delta s.t. X+delta in valid input range
        self.delta.data = torch.max(self.B[0]-xb,
                                    torch.min(self.B[1]-xb,
                                              self.delta.data))
        return self.delta

    def _get_loss(self, xb, yb, model, get_scores=False):
        out = model(xb+self.delta)

        if self.loss_type == 'untargeted':
            L = -1*F.cross_entropy(out, yb)

        elif self.loss_type == 'targeted':
            L = nn.CrossEntropyLoss()(out, yb)

        elif self.loss_type == 'random_targeted':
            rand_yb = torch.randint(low=0, high=self.num_classes, size=(len(yb),), device=self.device)
            #rand_yb[rand_yb==yb] = (yb[rand_yb==yb]+1) % self.num_classes
            L = nn.CrossEntropyLoss()(out, rand_yb)

        elif self.loss_type == 'plusone_targeted':
            next_yb = (yb+1) % self.num_classes
            L = nn.CrossEntropyLoss()(out, next_yb)

        elif self.loss_type == 'binary_targeted':
            yb_opp = 1-yb
            L = nn.CrossEntropyLoss()(out, yb_opp)

        elif self.loss_type ==  'binary_hybrid':
            yb_opp = 1-yb
            L = nn.CrossEntropyLoss()(out, yb_opp) - nn.CrossEntropyLoss()(out, yb)

        else:
            assert False, "unknown loss type"

        if get_scores: return L, out
        return L 

class L2_PGD_Attack(PGD_Attack):

    OVERFLOW_CONST = 1e-10

    def get_norms(self, X):
        nch = len(X.shape)
        return X.view(X.shape[0], -1).norm(dim=1)[(...,) + (None,)*(nch-1)]

    def _update_delta(self, xb, yb, update_mask=None):
        # normalize gradients
        grad = self.delta.grad.detach()
        norms = self.get_norms(grad)
        grad = grad/(norms+self.OVERFLOW_CONST) # add const to avoid overflow

        # steepest descent
        if self.minimal and update_mask is not None:
            um = update_mask
            self.delta.data[um] = self.delta.data[um] - self.lr*grad[um]
        else:
            self.delta.data = self.delta.data - self.lr*grad

        # l2 ball projection
        if self.project:
            delta_norms = self.get_norms(self.delta.data)
            self.delta.data = self.eps*self.delta.data / (delta_norms.clamp(min=self.eps))

        self.delta.grad.zero_()
        return self.delta

    def _init_delta(self, xb, yb):
        # random vector with L2 norm rand_eps
        delta = torch.zeros_like(xb)
        delta = delta.uniform_(-self.rand_eps, self.rand_eps)
        delta_norm = self.get_norms(delta)
        delta = self.rand_eps*delta/(delta_norm+self.OVERFLOW_CONST)
        delta = delta.to(self.device)
        delta.requires_grad = True
        return delta

class Linf_PGD_Attack(PGD_Attack):

    def _update_delta(self, xb, yb, **kw):
        # steepest descent + linf projection (GD)
        self.delta.data = self.delta.data - self.lr*(self.delta.grad.detach().sign())
        self.delta.data = self.delta.data.clamp(-self.eps, self.eps)
        self.delta.grad.zero_()
        return self.delta

