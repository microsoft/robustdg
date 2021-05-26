import seaborn as sns
import utils.scripts.gpu_utils as gu
import utils.scripts.data_utils as du
import utils.scripts.utils as utils
import random
import os, copy, pickle, time
import itertools
from collections import defaultdict, Counter, OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
#import foolbox
from torch.utils.data import TensorDataset, DataLoader
from torch import optim, nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

def parse_data(exps=None, root='/', **funcs_kw):
    """
    main function (parse data files and run added functions on it)
    """
    exps = exps if exps is not None else os.listdir(root)
    total = len(exps)
    print ("total: {}".format(total))
    parsed = defaultdict(dict)
    if total == 0: return parsed
    for idx, exp in enumerate(exps):
        if (idx+1) % 1 == 0: print (idx+1, end=' ', flush=True)
        # load data
        fpath = os.path.join(root, exp)
        try:
            data = torch.load(fpath, map_location=lambda storage, loc: storage)
        except:
            print ("File {} corrupted, skip.".format(fpath))
            continue
        config = data['config']

        # make exp config key
        config['run'] = int(exp.rsplit('.', 1)[0][-1])
        config['fname'] = exp
        ckeys = ['exp_name', 'dim', 'num_train', 'lin_margin', 'slab_margin', 'num_slabs',
                 'num_slabs', 'width', 'hdim', 'hl', 'linear', 'use_bn', 'run', 'fname',
                 'weight_decay', 'dropout']
        ckeys = [c for c in ckeys if c in config]
        cvals = [config[k] for k in ckeys]
        ckv = tuple(zip(ckeys, cvals))

        # save config
        parsed[ckv]['config'] = config

        # save functions
        for func_name, func in funcs_kw.items():
            parsed[ckv][func_name] = func(data)

    return parsed

def parse_exp_stats(data):
    """training summary statistics"""
    stats = data['stats']
    s = {}

    # loss + accuracy
    for t1, t2 in itertools.product(['acc', 'loss'], ['tr', 'te']):
        s['{}_{}'.format(t1,t2)] = stats['{}_{}'.format(t1, t2)][-1]
    s['orig_stats'] = stats
    s['acc_gap'] = s['acc_tr']-s['acc_te']
    s['loss_gap'] = s['loss_te']-s['loss_tr']
    s['fin_acc_te'] = stats['acc_te'][-1]
    s['fin_acc_tr'] = stats['acc_tr'][-1]

    # updates
    s['update_gap'] = stats['update_gap']
    s['num_updates'] = stats['num_updates']

    # effective number of updates
    for acc_threshold in [0.96, 0.97, 0.98, 0.99, 1]:
        eff = np.argmin(np.abs(np.array(stats['acc_tr'])-acc_threshold))*s['update_gap']
        s['effective_num_updates{}'.format(int(acc_threshold*100))] = eff

    return s

def parse_exp_model(data):
    """model parameter stats"""
    depth = data['config']['hl']
    linear = data['config']['linear']
    mtype = data['config'].get('mtype', 'fcn')
    if mtype == 'fcn' and depth == 1 and not linear: d = parse_exp_depth1_model(data)
    if mtype == 'fcn' and depth == 1 and linear: d = parse_exp_linear_model(data)
    return {}

def parse_exp_depth1_model(data):
    """cosine + w2"""
    device = gu.get_device()
    model = data['model'].to(device)
    p = W1, b1, w2, b2 = list(map(lambda x: x.detach().numpy(), model.parameters()))
    s = {}
    s['params'] = p
    s['cosine'] = W1[:, 0]/np.linalg.norm(W1, axis=1)
    s['l2'] = np.linalg.norm(W1, axis=1)
    s['w2'] = w2
    s['corr0'] = np.corrcoef(s['cosine'], w2[0, :])[0,1]
    s['corr1'] = np.corrcoef(s['cosine'], w2[1, :])[0,1]
    s['max_weight_cosine'] = s['cosine'][np.argmax(s['w2'][1,:])]
    return s

def parse_exp_linear_model(data):
    """cosine"""
    device = gu.get_device()
    model = data['model'].to(device)
    p = W,b = list(map(lambda x: x.detach().numpy(), model.parameters()))
    s = {}
    s['cosine0'], s['cosine1'] = W[:, 0]/np.linalg.norm(W, axis=1)
    return s

def parse_exp_data(data, load_X=False):
    s = {}
    model = data['model'].to(gu.get_device())
    data = data['data']
    X, Y = data['X'], data['Y']
    
    if type(X) != np.ndarray:
        X = data['X'].detach().cpu()
        
    if type(X) != np.ndarray:
        Y = data['Y'].detach().cpu()
    
    s['Y'] = Y
    if load_X: s['X'] = X
    s['Y_'] = get_yhat(model, X)
    s['model'] = model
    return s

def get_yhat(model, data):
    if type(data)==np.ndarray: data = torch.Tensor(data)
    return torch.argmax(model(data), 1)

def get_acc(y,yhat):
    n = float(len(y))
    return (y==yhat).sum().item()/n

def parse_and_get_df(root, prefix, files=None, device_id=None, only_load=False, only_linear=False, sample_pct=0.5, load_X=False, use_model_pred=False):
    exps = files if files is not None else [f for f in os.listdir(root) if f.startswith(prefix)]

    funcs = {
        'config': lambda d: d['config'],
        'stats': parse_exp_stats,
        'model': parse_exp_model,
        'data': lambda x: parse_exp_data(x, load_X=load_X),
        'random_dep': lambda d: get_feature_deps(d['data']['te_dl'], d['model'], only_linear=only_linear, W=d['data'].get('W', None), dep_type='random', use_model_pred=use_model_pred, print_info=False, sample_pct=sample_pct, device_id=device_id),
        'swap_dep': lambda d: get_feature_deps(d['data']['te_dl'], d['model'], only_linear=only_linear, W=d['data'].get('W', None), dep_type='swap', use_model_pred=use_model_pred, print_info=False, sample_pct=sample_pct, device_id=device_id),
    }

    P = parse_data(root=root, exps=exps, **funcs)
    if only_load: return P

    D = []
    for idx, (k,v) in enumerate(P.items(),1):
        d = OrderedDict()
        for a,b in k: d[a] = b
        for vk in ['model', 'data', 'stats', 'config']:
            for a,b in v[vk].items(): d[a] = b
        for vk in ['random_dep', 'swap_dep']:
            for coord, dep in v[vk].items():
                d[f'{vk[0]}dep_{coord}'] = dep
        D.append(d)

    df = pd.DataFrame(D)
    if len(df): df['nd'] = df['num_train']/df['dim']
    return df

def viz(d, c1, c2, k=80_000, info=True, plot_dm=True, plot_data=True, use_yhat=False, unif_k=False, width=10, title=None, is_binary=False, dep_type='swap', ax=None):
    if 'W' not in d['data']: W = np.eye(d['config']['dim'])
    else: W = d['data']['W']
    if W is None: W = np.eye(d['config']['dim'])

    z = parse_exp_data(d)
    X = d['data']['X']

    # visualize un-transformed data...
    X_ = np.array(X).dot(W.T)
    Y, Y_ = z['Y'], z['Y_']
    model = d['model'].cpu()
    D = X.shape[1]
    kn = k if unif_k else len(X)
    K = torch.Tensor(np.random.uniform(size=(k, D)))*width if unif_k else np.array(X_)
    K[:, c1] = torch.Tensor(np.random.uniform(low=min(X_[:,c1]), high=max(X_[:,c1]), size=kn))
    K[:, c2] = torch.Tensor(np.random.uniform(low=min(X_[:,c2]), high=max(X_[:,c2]), size=kn))
    KO = model(torch.Tensor(np.array(K).dot(W)))
    if is_binary: KY = (KO > 0).squeeze().numpy()
    else: KY = torch.argmax(KO, 1).numpy()

    if info:
        deps = get_feature_deps(d['data']['te_dl'], d['model'], W=d['data'].get('W', None), dep_type=dep_type)
        for k,v in sorted(deps.items(), reverse=False, key=lambda t: t[-1]): print ('{}:{:.3f}'.format(k,v), end=' ')
        print ("\n")

    if ax is None: fig, ax = plt.subplots(1,1,figsize=(6,4))

    if plot_dm: ax.scatter(K[:, c1], K[:, c2], c=KY, cmap='binary', s=8, alpha=.2)
    if plot_data: ax.scatter(X_[:, c1], X_[:, c2], c=Y_ if use_yhat else Y, cmap='coolwarm', s=8, alpha=.4)

    ax.set_xlabel('e_{}'.format(c1))
    ax.set_ylabel('e_{}'.format(c2))
    ax.set_title(title if title else '')
    plt.tight_layout()
    return ax

def visualize_boundary(model, data, c1, c2, dim, ax=None, is_binary=False, use_yhat=False, width=1, unif_k=True, k=100_000, print_info=True, dep_type='random'):
    agg = {'model': model, 'data': data, 'config': dict(dim=dim)}
    return viz(agg, c1, c2, unif_k=unif_k, width=width, dep_type=dep_type, is_binary=is_binary, use_yhat=use_yhat, ax=ax, info=print_info)

def get_randomized_loader(dl, W, coordinates):
    """
    dl: dataloader
    W: rotation matrix
    coordinates: list of coordinates to randomize
    output: randomized dataloader
    """

    def _randomize(X, coords):
        p = torch.randperm(len(X))
        for c in coords: X[:, c] = X[p, c]
        return X

    # rotate data
    X, Y = map(copy.deepcopy, dl.dataset.tensors)
    dim = X.shape[1]
    if W is None: W = np.eye(dim)

    rt_X = torch.Tensor(X.numpy().dot(W.T))
    rand_rt_X = _randomize(rt_X, coordinates)
    rand_X = torch.Tensor(rand_rt_X.numpy().dot(W))

    return utils._to_dl(rand_X, Y, dl.batch_size)


def get_feature_deps(dl, model, W=None, dep_type='random', only_linear=False, coords=None, metric='accuracy', 
                     use_model_pred=False, print_info=False, sample_pct=1.0, device_id=None):
    """Compute feature dependencies using randomization or swapping"""
    def _randomize(X, Y, coords):
        p = torch.randperm(len(X))
        for c in coords: X[:, c] = X[p, c]
        return X

    def _swap(X, Y, coords):
        idx0, idx1 = map(lambda c: (Y.numpy()==c).nonzero()[0], [0, 1])
        idx0_new = np.random.choice(idx1, size=len(idx0), replace=True)
        idx1_new = np.random.choice(idx0, size=len(idx1), replace=True)
        for c in coords: X[idx0, c], X[idx1, c] = X[idx0_new, c], X[idx1_new, c]
        return X

    def _get_dep_data(X, Y, coords):
        return dict(random=_randomize, swap=_swap)[dep_type](X, Y, coords)


    assert metric in {'accuracy', 'loss', 'auc'}

    # setup data
    device = gu.get_device(device_id)
    model = model.to(device)
    X, Y = map(lambda Z: Z.to(device), dl.dataset.tensors)
    Yh = get_yhat(model, X)
    dim = X.shape[1]
    if W is None: W = np.eye(dim)
    W = torch.Tensor(W).to(device)
    rt_X = torch.mm(X, torch.transpose(W,0,1))

    # subsample data
    n_samp = int(round(sample_pct*len(rt_X)))
    perm = torch.randperm(len(rt_X))[:n_samp]
    rt_X, Y, Yh = rt_X[perm, :], Y[perm], Yh[perm]

    # compute deps
    deps = {}

    dims = list(range(dim))
    if coords is None and not only_linear: coords = dims
    if coords is None and only_linear: coords = [0,1]

    for idx, coord in enumerate(coords):
        if print_info: print ('{}/{}'.format(idx, len(coords)), end=' ')
        rt_X_ = copy.deepcopy(rt_X).to(device)
        rt_X_ = _get_dep_data(rt_X_, Y, coord if type(coord) in (list, tuple) else [coord])
        X_ = torch.mm(rt_X_, W)
        Ys = get_yhat(model, X_)

        key = tuple(coord) if type(coord) in (list, tuple) else coord

        if metric == 'auc':
            L = utils.get_logits_given_tensor(X_, model, device=device, bs=250)
            S = L[:,1]-L[:,0]
            auc = roc_auc_score(Y.cpu().numpy(), S.cpu().numpy())
            deps[key] = auc
        elif metric == 'accuracy':
            deps[key] = get_acc(Yh if use_model_pred else Y, Ys)
        elif metric == 'loss':
            L = utils.get_logits_given_tensor(X_, model, device=device, bs=250)
            with torch.no_grad():
                loss_val = F.cross_entropy(L, Y).item()
            deps[key] = loss_val

    return deps

def get_subset_feature_deps(dl, model, coords_set, comb_size, W=None, dep_type='random', sample_pct=0.5, device_id=None, print_info=False):
    coords = list(itertools.combinations(coords_set, comb_size))
    return get_feature_deps(dl, model, W=W, dep_type=dep_type, coords=coords, print_info=print_info, sample_pct=sample_pct, device_id=device_id)
