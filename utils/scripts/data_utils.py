import sys

import random, os, copy, pickle, time, random, argparse, itertools
from collections import defaultdict, Counter, OrderedDict
import numpy as np
import torch
import torchvision
from torch import optim, nn
import torch.nn.functional as F
from sklearn import metrics
from torch.utils.data import TensorDataset, DataLoader

import utils.scripts.gpu_utils as gu
import utils.scripts.lms_utils as au
import utils.scripts.synth_models as synth_models
import utils.scripts.utils as utils
import matplotlib.pyplot as plt
import pathlib

try:
    sys.path.append('../../cifar10_models/')
    import cifar10_models as c10
    c10_not_found = False
except:
    c10_not_found = True

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

REPO_DIR = pathlib.Path(__file__).parent.parent.absolute()
DOWNLOAD_DIR = os.path.join(REPO_DIR, 'datasets')

def msd(x, r=3):
    return np.round(np.mean(x), r), np.round(np.std(x), r)

def _get_dataloaders(trd, ted,  bs, pm=True, shuffle=True):
    train_dl = DataLoader(trd, batch_size=bs, shuffle=shuffle, pin_memory=pm)
    test_dl = DataLoader(ted, batch_size=bs, pin_memory=pm)
    return train_dl, test_dl

def get_cifar10_models(device=None, pretrained=True):
    if c10_not_found: return {}
    device = gu.get_device(None) if device is None else device
    get_lmbda = lambda cls: (lambda: cls(pretrained=pretrained).eval().to(device))
    return {
        'vgg11_bn': get_lmbda(c10.vgg11_bn),
        'vgg13_bn': get_lmbda(c10.vgg13_bn),
        'vgg16_bn': get_lmbda(c10.vgg16_bn),
        'vgg19_bn': get_lmbda(c10.vgg19_bn),
        'resnet18': get_lmbda(c10.resnet18),
        'resnet34': get_lmbda(c10.resnet34),
        'resnet50': get_lmbda(c10.resnet50),
        'densenet121': get_lmbda(c10.densenet121),
        'densenet161': get_lmbda(c10.densenet161),
        'densenet169': get_lmbda(c10.densenet169),
        'mobilenet_v2': get_lmbda(c10.mobilenet_v2),
        'googlenet': get_lmbda(c10.googlenet),
        'inception_v3': get_lmbda(c10.inception_v3)
    }

def plot_decision_boundary(dl, model, c1, c2, ax=None, print_info=True):
    if ax is None: fig, ax = plt.subplots(1,1,figsize=(6,4))
    model = model.cpu()
    deps = sorted(au.get_feature_deps(dl, model).items(), key=lambda t: t[-1])

    if print_info:
        for k, v in deps: print ('{}:{:.3f}'.format(k,v), end=', ')
        print ("")
        
    X, Y = utils.extract_numpy_from_loader(dl)
    K = 100_000
    U = np.random.uniform(low=X.min(), high=X.max(), size=(K, X.shape[1])) # copy.deepcopy(X)
    U[:, c1] = np.random.uniform(low=X[:, c1].min(), high=X[:, c1].max(), size=K)
    U[:, c2] = np.random.uniform(low=X[:, c2].min(), high=X[:, c2].max(), size=K)
    U = torch.Tensor(U)
    
    with torch.no_grad():
        out = model(U)
        Yu = torch.argmax(out, 1)        
        
    ax.scatter(U[:,c1], U[:,c2], c=Yu, alpha=0.3, s=24)
    ax.scatter(X[:,c1], X[:,c2], c=Y, cmap='coolwarm', s=12)

def get_binary_datasets(X, Y, y1, y2, image_width=28, use_cnn=False):
    assert type(X) is np.ndarray and type(Y) is np.ndarray
    idx0 = (Y==y1).nonzero()[0]
    idx1 = (Y==y2).nonzero()[0]
    idx = np.concatenate((idx0, idx1))
    X_, Y_ = X[idx,:], (Y[idx]==y2).astype(int)
    P = np.random.permutation(len(X_))
    X_, Y_ = X_[P,:], Y_[P]
    if use_cnn: X_ = X_.reshape(X.shape[0], -1, image_width)[:, None, :, :]
    return X_[P,:], Y_[P]

def get_binary_loader(dl, y1, y2):
    X, Y = utils.extract_numpy_from_loader(dl)
    X, Y = get_binary_datasets(X, Y, y1, y2)
    return utils._to_dl(X, Y, bs=dl.batch_size)

def get_mnist(fpath=DOWNLOAD_DIR, flatten=False, binarize=False, normalize=True, y0={0,1,2,3,4}):
    """get preprocessed mnist torch.TensorDataset class"""
    def _to_torch(d):
        X, Y = [], []
        for xb, yb in d:
            X.append(xb)
            Y.append(yb)
        return torch.Tensor(np.stack(X)), torch.LongTensor(np.stack(Y))

    to_tensor = torchvision.transforms.ToTensor()
    to_flat = torchvision.transforms.Lambda(lambda X: X.reshape(-1).squeeze())
    to_norm = torchvision.transforms.Normalize((0.5, ), (0.5, ))
    to_binary = torchvision.transforms.Lambda(lambda y: 0 if y in y0 else 1)

    transforms = [to_tensor]
    if normalize: transforms.append(to_norm)
    if flatten: transforms.append(to_flat)
    tf = torchvision.transforms.Compose(transforms)
    ttf = to_binary if binarize else None

    X_tr = torchvision.datasets.MNIST(fpath, download=True, transform=tf, target_transform=ttf)
    X_te = torchvision.datasets.MNIST(fpath, download=True, train=False, transform=tf, target_transform=ttf)

    return _to_torch(X_tr), _to_torch(X_te)

def get_mnist_dl(fpath=DOWNLOAD_DIR, to_np=False, bs=128, pm=False, shuffle=False,
                 normalize=True, flatten=False, binarize=False, y0={0,1,2,3,4}):
    (X_tr, Y_tr), (X_te, Y_te) = get_mnist(fpath, normalize=normalize, flatten=flatten, binarize=binarize, y0=y0)
    tr_dl = DataLoader(TensorDataset(X_tr, Y_tr), batch_size=bs, shuffle=shuffle, pin_memory=pm)
    te_dl = DataLoader(TensorDataset(X_te, Y_te), batch_size=bs, pin_memory=pm)
    return tr_dl, te_dl

def get_cifar(fpath=DOWNLOAD_DIR, use_cifar10=False, flatten_data=False, transform_type='none',
              means=None, std=None, use_grayscale=False, binarize=False, normalize=True, y0={0,1,2,3,4}):
    """get preprocessed cifar torch.Dataset class"""

    if transform_type == 'none':
        normalize_cifar = lambda: torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        tensorize = torchvision.transforms.ToTensor()
        to_grayscale = torchvision.transforms.Grayscale()
        flatten = torchvision.transforms.Lambda(lambda X: X.reshape(-1).squeeze())

        transforms = [tensorize]
        if use_grayscale: transforms = [to_grayscale] + transforms
        if normalize: transforms.append(normalize_cifar())
        if flatten_data: transforms.append(flatten)
        tr_transforms = te_transforms = torchvision.transforms.Compose(transforms)

    if transform_type == 'basic':
        normalize_cifar = lambda: torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])

        tr_transforms= [
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor()
        ]

        te_transforms = [
            torchvision.transforms.Resize(32),
            torchvision.transforms.CenterCrop(32),
            torchvision.transforms.ToTensor(),
        ]

        if normalize:
            tr_transforms.append(normalize_cifar())
            te_transforms.append(normalize_cifar())

        tr_transforms = torchvision.transforms.Compose(tr_transforms)
        te_transforms = torchvision.transforms.Compose(te_transforms)

    to_binary = torchvision.transforms.Lambda(lambda y: 0 if y in y0 else 1)
    target_transforms = to_binary if binarize else None
    dset = 'cifar10' if use_cifar10 else 'cifar100'
    func = torchvision.datasets.CIFAR10 if use_cifar10 else torchvision.datasets.CIFAR100

    X_tr = func(fpath, download=True, transform=tr_transforms, target_transform=target_transforms)
    X_te = func(fpath, download=True, train=False, transform=te_transforms, target_transform=target_transforms)

    return X_tr, X_te

def get_cifar_dl(fpath=DOWNLOAD_DIR, use_cifar10=False, bs=128, shuffle=True, transform_type='none',
                 means=None, std=None, normalize=True, flatten_data=False, use_grayscale=False, nw=4, pm=False, binarize=False, y0={0,1,2,3,4}):
    """data in dataloaders have has shape (B, C, W, H)"""
    d_tr, d_te = get_cifar(fpath, use_cifar10=use_cifar10, use_grayscale=use_grayscale, transform_type=transform_type, normalize=normalize, means=means, std=std, flatten_data=flatten_data, binarize=binarize, y0=y0)
    tr_dl = DataLoader(d_tr, batch_size=bs, shuffle=shuffle, num_workers=nw, pin_memory=pm)
    te_dl = DataLoader(d_te, batch_size=bs, num_workers=nw, pin_memory=pm)
    return tr_dl, te_dl

def get_cifar_np(fpath=DOWNLOAD_DIR, use_cifar10=False, flatten_data=False, transform_type='none', normalize=True, binarize=False, y0={0,1,2,3,4}, use_grayscale=False):
    """get numpy matrices of preprocessed cifar data"""

    def _to_np(d):
        X, Y = [], []
        for xb, yb in d:
            X.append(xb)
            Y.append(yb)
        return map(np.stack, [X,Y])

    d_tr, d_te = get_cifar(fpath, use_cifar10=use_cifar10, use_grayscale=use_grayscale, transform_type=transform_type, normalize=normalize, flatten_data=flatten_data, binarize=binarize, y0=y0)
    return _to_np(d_tr), _to_np(d_te)

if __name__ == '__main__':
    pass