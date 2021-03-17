import random
import os, copy, pickle, time
import itertools
from collections import defaultdict, Counter, OrderedDict
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import utils
import gpu_utils as gu
import data_utils as du

def get_binary_mnist(y1=0, y2=1, apply_padding=True, repeat_channels=True):
    
    def _make_cifar_compatible(X):
        if apply_padding: X = np.stack([np.pad(X[i][0], 2)[None,:] for i in range(len(X))]) # pad
        if repeat_channels: X = np.repeat(X, 3, axis=1) # add channels
        return X

    binarize = lambda X,Y: du.get_binary_datasets(X, Y, y1=y1, y2=y2)
    
    tr_dl, te_dl = du.get_mnist_dl(normalize=False)
    Xtr, Ytr = binarize(*utils.extract_numpy_from_loader(tr_dl))
    Xte, Yte = binarize(*utils.extract_numpy_from_loader(te_dl))
    Xtr, Xte = map(_make_cifar_compatible, [Xtr, Xte])
    return (Xtr, Ytr), (Xte, Yte)

def get_binary_cifar(y1=3, y2=5, c={0,1,2,3,4}, use_cifar10=True):
    binarize = lambda X,Y: du.get_binary_datasets(X, Y, y1=y1, y2=y2)
    binary = False if y1 is not None and y2 is not None else True
    if binary: print ("grouping cifar classes")
    tr_dl, te_dl = du.get_cifar_dl(use_cifar10=use_cifar10, shuffle=False, normalize=False, binarize=binary, y0=c)

    Xtr, Ytr = binarize(*utils.extract_numpy_from_loader(tr_dl))
    Xte, Yte = binarize(*utils.extract_numpy_from_loader(te_dl))
    return (Xtr, Ytr), (Xte, Yte)

def combine_datasets(Xm, Ym, Xc, Yc, randomize_order=False, randomize_first_block=False, randomize_second_block=False):
    """combine two datasets"""

    def partition(X, Y, randomize=False):
        """partition randomly or using labels"""
        if randomize: 
            n = len(Y)
            p = np.random.permutation(n)
            ni, pi = p[:n//2], p[n//2:]
        else:
            ni, pi = (Y==0).nonzero()[0], (Y==1).nonzero()[0]
        return X[pi], X[ni]

    def _combine(X1, X2):
        """concatenate images from two sources"""
        X = []
        for i in range(min(len(X1), len(X2))):
            x1, x2 = X1[i], X2[i]
            # randomize order 
            if randomize_order and random.random() < 0.5:
                x1, x2 = x2, x1
            x = np.concatenate((x1,x2), axis=1)
            X.append(x)
        return np.stack(X)

    Xmp, Xmn = partition(Xm, Ym, randomize=randomize_first_block)
    Xcp, Xcn = partition(Xc, Yc, randomize=randomize_second_block)
    n = min(map(len, [Xmp, Xmn, Xcp, Xcn]))
    Xmp, Xmn, Xcp, Xcn = map(lambda Z: Z[:n], [Xmp, Xmn, Xcp, Xcn])

    Xp = _combine(Xmp, Xcp)
    Yp = np.ones(len(Xp))

    Xn = _combine(Xmn, Xcn)
    Yn = np.zeros(len(Xn))
    
    X = np.concatenate([Xp, Xn], axis=0)
    Y = np.concatenate([Yp, Yn], axis=0)
    P = np.random.permutation(len(X))
    X, Y = X[P], Y[P]
    return X, Y

def get_mnist_cifar(mnist_classes=(0,1), cifar_classes=None, c={0,1,2,3,4}, 
                    randomize_mnist=False, randomize_cifar=False):  
        
    y1, y2 = mnist_classes
    (Xtrm, Ytrm), (Xtem, Ytem) = get_binary_mnist(y1=y1, y2=y2)
    
    y1, y2 = (None, None) if cifar_classes is None else cifar_classes
    (Xtrc, Ytrc), (Xtec, Ytec) = get_binary_cifar(c=c, y1=y1, y2=y2)
    
    Xtr, Ytr = combine_datasets(Xtrm, Ytrm, Xtrc, Ytrc, randomize_first_block=randomize_mnist, randomize_second_block=randomize_cifar)
    Xte, Yte = combine_datasets(Xtem, Ytem, Xtec, Ytec, randomize_first_block=randomize_mnist, randomize_second_block=randomize_cifar)
    return (Xtr, Ytr), (Xte, Yte)

def get_mnist_cifar_dl(mnist_classes=(0,1), cifar_classes=None, c={0,1,2,3,4}, bs=256, 
                       randomize_mnist=False, randomize_cifar=False):
    (Xtr, Ytr), (Xte, Yte) = get_mnist_cifar(mnist_classes=mnist_classes, cifar_classes=cifar_classes, 
                                             c=c, randomize_mnist=randomize_mnist, randomize_cifar=randomize_cifar)
    tr_dl = utils._to_dl(Xtr, Ytr, bs=bs, shuffle=True)
    te_dl = utils._to_dl(Xte, Yte, bs=100, shuffle=False)
    return tr_dl, te_dl