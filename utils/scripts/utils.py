import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import copy
from collections import defaultdict, Counter, OrderedDict
import time
from torch.utils.data import TensorDataset, DataLoader
import torchvision
from torch import optim, nn
import torch.nn.functional as F
from scipy.linalg import qr
import utils.scripts.lms_utils as au
import utils.scripts.ptb_utils as pu
import utils.scripts.gpu_utils as gu
from sklearn import metrics
import collections
from sklearn.metrics import roc_auc_score

plt.style.use('seaborn-ticks')
import matplotlib.ticker as ticker

def get_orthonormal_matrix(n):
    H = np.random.randn(n, n)
    s = np.linalg.svd(H)[1]
    s = s[s>1e-7]
    if len(s) != n: return get_orthonormal_matrix(n)
    Q, R = qr(H)
    return Q

def get_dataloader(X, Y, bs, **kw):
    return DataLoader(TensorDataset(X, Y), batch_size=bs, **kw)

def split_dataloader(dl, frac=0.5):
    bs = dl.batch_size
    X, Y = dl.dataset.tensors
    p = torch.randperm(len(X))
    X, Y = X[p, :], Y[p]
    n = int(round(len(X)*frac))
    X0, Y0 = X[:n, :], Y[:n]
    X1, Y1 = X[n:, :], Y[n:]
    dl0 = DataLoader(TensorDataset(torch.Tensor(X0), torch.LongTensor(Y0)), batch_size=bs, shuffle=True)
    dl1 = DataLoader(TensorDataset(torch.Tensor(X1), torch.LongTensor(Y1)), batch_size=bs, shuffle=True)
    return  dl0, dl1

def _to_dl(X, Y, bs, shuffle=True):
    return DataLoader(TensorDataset(torch.Tensor(X), torch.LongTensor(Y)), batch_size=bs, shuffle=shuffle)

def extract_tensors_from_loader(dl, repeat=1, transform_fn=None):
    X, Y = [], []
    for _ in range(repeat):
        for xb, yb in dl:
            if transform_fn:
                xb, yb = transform_fn(xb, yb)
            X.append(xb)
            Y.append(yb)
    X = torch.FloatTensor(torch.cat(X))
    Y = torch.LongTensor(torch.cat(Y))
    return X, Y

def extract_numpy_from_loader(dl, repeat=1, transform_fn=None):
    X, Y = extract_tensors_from_loader(dl, repeat=repeat, transform_fn=transform_fn)
    return X.numpy(), Y.numpy()

def _to_tensor_dl(dl, repeat=1, bs=None):
    X, Y = extract_numpy_from_loader(dl, repeat=repeat)
    dl = _to_dl(X, Y, bs if bs else dl.batch_size)
    return dl

def flatten_loader(dl, bs=None):
    X, Y = extract_numpy_from_loader(dl)
    X = X.reshape(X.shape[0], -1)
    return _to_dl(X, Y, bs=bs if bs else dl.batch_size)

def merge_loaders(dla, dlb):
    bs = dla.batch_size
    Xa, Ya = extract_numpy_from_loader(dla)
    Xb, Yb = extract_numpy_from_loader(dlb)
    return _to_dl(np.concatenate([Xa, Xb]), np.concatenate([Ya, Yb]), bs)

def transform_loader(dl, func, shuffle=True):
    #assert type(dl.sampler) is torch.utils.data.sampler.SequentialSampler
    X, Y = extract_numpy_from_loader(dl, transform_fn=func)
    return _to_dl(X, Y, dl.batch_size, shuffle=shuffle)

def visualize_tensors(P, size=8, normalize=True, scale_each=False, permute=True, ax=None, pad_value=0.):
    if ax is None: _, ax = plt.subplots(1,1,figsize=(20,4))
    if permute:
        s = np.random.choice(len(P), size=size, replace=False)
        p = P[s]
    else:
        p = P[:size]
    g = torchvision.utils.make_grid(torch.FloatTensor(p), nrow=size, normalize=normalize, scale_each=scale_each, pad_value=pad_value)
    g = g.permute(1,2,0).numpy()
    ax.imshow(g)
    ax.set_xticks([])
    ax.set_yticks([])
    return ax
    
def visualize_loader(dl, ax=None, size=8, normalize=True, scale_each=False, reshape=None): 
    if ax is None: _, ax = plt.subplots(1,1,figsize=(20,4))
    for xb, yb in dl: break
    if reshape: xb = xb.reshape(len(xb), *reshape)
    return visualize_tensors(xb, size=size, normalize=normalize, scale_each=scale_each, permute=True, ax=ax)

def visualize_loader_by_class(dl, ax=None, size=8, normalize=True, scale_each=False, reshape=None): 
    for xb, yb in dl: break
    if reshape: xb = xb.reshape(len(xb), *reshape)

    classes = list(set(list(yb.numpy())))
    fig, axs = plt.subplots(len(classes), 1, figsize=(15, 3*len(classes)))

    for y, ax in zip(classes, axs):
        xb_ = xb[yb==y]
        ax = visualize_tensors(xb_, size=size, normalize=normalize, scale_each=scale_each, permute=True, ax=ax)
        ax.set_title('Class: {}'.format(y))

    return fig
    
def visualize_perturbations(P, transform_fn=None):
    if transform_fn is not None:
        P = transform_fn(P)
    plt.figure(figsize=(20,4))
    s = np.random.choice(len(P), size=8, replace=False)
    p = P[s]
    g = torchvision.utils.make_grid(torch.FloatTensor(p))
    g = g.permute(1,2,0).numpy()
    g = (g-g.min())/g.max()
    plt.imshow(g)

def get_logits_given_tensor(X, model, device=None, bs=250, softmax=False):
    if device is None: device = gu.get_device(None)
    sampler = torch.utils.data.SequentialSampler(X)
    sampler = torch.utils.data.BatchSampler(sampler, bs, False)
    
    logits = []
    
    with torch.no_grad():
        model = model.to(device)
        for idx in sampler:
            xb = X[idx].to(device)
            out = model(xb)
            logits.append(out)
            
    L = torch.cat(logits)
    if softmax: return F.softmax(L, 1)
    return L

def get_predictions_given_tensor(X, model, device=None, bs=250):
    out = get_logits_given_tensor(X, model, device=device, bs=bs)
    return torch.argmax(out, 1)

def get_accuracy_given_tensor(X, Y, model, device=None, bs=250):
    if device is None: device = gu.get_device(None)
    Y = torch.LongTensor(Y).to(device)
    yhat = get_predictions_given_tensor(X, model, device=device, bs=bs)
    return (Y==yhat).float().mean().item()

def compute_accuracy(X, Y, model):
    with torch.no_grad():
        pred = torch.argmax(model(X),1)
        correct = (pred == Y).sum().item()
        accuracy = correct/float(len(Y))
    return accuracy

def compute_loss_and_accuracy_from_dl(dl, model, loss_fn, sample_pct=1.0, device=None, transform_fn=None):
    in_tr_mode = model.training
    model = model.eval()
    data_size = float(len(dl.dataset))
    samp_size = int(np.ceil(sample_pct*data_size))
    num_eval = 0.
    bs = dl.batch_size
    accs, losses, bss = [], [], []

    with torch.no_grad():
        for xb, yb in dl:
            xb, yb = xb.to(device, non_blocking=False), yb.to(device, non_blocking=False)

            if transform_fn:
                xb, yb = transform_fn(xb, yb)

            sc = model(xb)

            if loss_fn is F.cross_entropy:
                loss = loss_fn(sc, yb, reduction='mean')
                pred = torch.argmax(sc, 1)
            elif loss_fn is F.binary_cross_entropy_with_logits:
                loss = loss_fn(sc, yb.float().unsqueeze(1))
                pred = (sc > 0.).long().squeeze()
            elif loss_fn is hinge_loss:
                loss = loss_fn(sc, yb)
                pred = (sc > 0).long().squeeze()
            else:
                try:
                    loss = loss_fn(sc, yb)
                    pred = torch.argmax(sc, 1)
                except:
                    assert False, "unknown loss function"

            correct = (pred==yb).sum().float()
            n = float(len(xb))
            losses.append(loss)
            accs.append(correct/n)
            bss.append(n)

            num_eval += n
            if num_eval >= samp_size: break

    accs, losses, bss = map(np.array, [accs, losses, bss])
    if in_tr_mode: model = model.train()
    return np.sum(bss*accs)/num_eval, np.sum(bs*losses)/num_eval

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_logits(model, loader, device):
    S, Y = [], []
    with torch.no_grad():
        for xb, yb in loader:
            #print(xb, yb)
            xb = xb.to(device)
            out = model(xb).cpu().numpy()
            S.append(out)
            Y.append(list(yb))
    S, Y = map(np.concatenate, [S, Y])
    return S, Y

def get_scores(model, loader, device):
    """binary tasks only"""
    S, Y = get_logits(model, loader, device)
    print('Acc check: ', 100*np.sum(np.argmax(S, axis=1)==Y)/Y.shape[0])
    return S[:,1]-S[:,0], Y

def get_multiclass_logit_score(L, Y):
    scores = [] 
    for idx, (l, y) in enumerate(zip(L, Y)):
        sc_y = l[y]
        
        indices = np.argsort(l)
        best2_idx, best1_idx = indices[-2:]
        sc_max = l[best2_idx] if y == best1_idx else l[best1_idx]
        
        score = sc_y - sc_max
        scores.append(score)
        
    return np.array(scores)

def get_binary_auc(model, loader, device):
    S, Y = get_scores(model, loader, device)
    return roc_auc_score(Y, S)

def get_multiclass_auc(model, loader, device, one_vs_rest=True):
    X, Y = extract_tensors_from_loader(loader)
    S = get_logits_given_tensor(X, model, device=device, softmax=True).cpu()
    mc = 'ovr' if one_vs_rest is True else 'ovo'
    S, Y = S.numpy(), Y.numpy()
    return roc_auc_score(Y, S, multi_class=mc)

def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params: p.grad.data.clamp_(-clip_value, clip_value)

def print_model_gradients(model, print_bias=True):
    for name, params in model.named_parameters():
        if not print_bias and 'bias' in name: continue
        if not params.requires_grad: continue
        avg_grad = np.mean(params.grad.cpu().numpy())
        print (name, params.shape, avg_grad)

def hinge_loss(out, y):
    y_ = (2*y.float()-1).unsqueeze(1)
    return torch.mean(F.relu(1-out*y_))

def pgd_adv_fit_model(model, opt, tr_dl, te_dl, attack, eval_attack=None, device=None, sch=None, max_epochs=100, epoch_gap=2, 
                      min_loss=0.001, print_info=True, save_init_model=True):
                    
    # setup tracking 
    PR = lambda x: print (x) if print_info else None
    stop_training = False
    stats = defaultdict(list)
    best_val, best_model = np.inf, None
    adv_epoch_timer = []
    epoch_gap_timer = [time.time()]
    init_model = copy.deepcopy(model).cpu() if save_init_model else None
    
    # eval attack 
    eval_attack = eval_attack or attack

    print ("Min loss: {}".format(min_loss))

    def standard_epoch(loader, model, optimizer=None, sch=None):
        """compute accuracy and loss. Backprop if optimizer provided"""
        total_loss, total_err = 0.,0.
        model = model.eval() if optimizer is None else model.train()
        model = model.to(device)
        update_params = optimizer is not None

        with torch.set_grad_enabled(update_params):
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                yp = model(xb)
                loss = F.cross_entropy(yp, yb)
                if update_params:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                total_err += (yp.max(dim=1)[1] != yb).sum().item()
                total_loss += loss.item() * xb.shape[0]
        return total_err / len(loader.dataset), total_loss / len(loader.dataset)

    def adv_epoch(loader, model, attack, optimizer=None, sch=None):
        """compute adv accuracy and loss. Backprop if optimizer provided"""
        start_time = time.time()
        total_loss, total_err = 0.,0.
        model = model.eval() if optimizer is None else model.train()
        model = model.to(device)
        update_params = optimizer is not None

        for xb, yb in loader:
            torch.set_grad_enabled(True)
            xb, yb = xb.to(device), yb.to(device)
            delta = attack.perturb(xb, yb, model).to(device)
            xb = xb + delta
            with torch.set_grad_enabled(update_params):
                yp = model(xb)
                loss = F.cross_entropy(yp, yb)
                if update_params:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            total_err += (yp.max(dim=1)[1] != yb).sum().item()
            total_loss += loss.item() * xb.shape[0]

        if optimizer is not None and sch is not None:
            cur_lr = next(iter(opt.param_groups))['lr']
            sch.step()
            new_lr = next(iter(opt.param_groups))['lr']
            if new_lr != cur_lr:
                PR('Epoch {}, LR : {} -> {}'.format(epoch, cur_lr, new_lr))

        total_time = time.time()-start_time
        adv_epoch_timer.append(total_time)
        return total_err / len(loader.dataset), total_loss / len(loader.dataset)

    epoch = 0
    while epoch < max_epochs:        
        if stop_training:
            break
        try:
            stat = {}
            model = model.train()
            train_err, train_loss = adv_epoch(tr_dl, model, attack, optimizer=opt, sch=sch)

            if epoch % epoch_gap == 0:
                model = model.eval()
                test_err, test_loss = standard_epoch(te_dl, model, optimizer=None, sch=None)
                adv_err, adv_loss = adv_epoch(te_dl, model, eval_attack, optimizer=None, sch=None)
                stat['acc_te'], stat['acc_te_std'] = adv_err, test_err
                stat['loss_te'], stat['loss_te_std'] = adv_loss, test_loss

                if adv_err < best_val:
                    best_val = adv_err
                    best_model = copy.deepcopy(model).eval()

                if print_info:
                    if epoch==0: print  ("Epoch", "l-tr", "a-tr", "a-te", "s-te", "time", sep='\t')
                    #print (epoch, *("{:.4f}".format(i) for i in (train_loss, train_err)), sep='   ')
                    diff_time = time.time()-epoch_gap_timer[-1]
                    epoch_gap_timer.append(time.time())
                    print (epoch, *("{:.4f}".format(i) for i in (train_loss, 1.-train_err, 1.-adv_err, 1.-test_err, diff_time)), sep='   ')

                if train_loss < min_loss:
                    stop_training = True

            print ("Epoch {}: accuracy {:.3f} and loss {:.3f}".format(epoch, 1-train_err, train_loss))

            stat['epoch'] = epoch
            stat['acc_tr'] = train_err
            stat['loss_tr'] = train_loss

            for k, v in stat.items():
                stats[k].append(v)

            epoch += 1

        except KeyboardInterrupt:
            inp = input("LR num or Q or SAVE or GAP or MAXEPOCHS: ")
            if inp.startswith('LR'):
                lr = float(inp.split(' ')[-1])
                cur_lr = next(iter(opt.param_groups))['lr']
                PR("New LR: {}".format(lr))
                for g in opt.param_groups: g['lr'] = lr
            if inp.startswith('Q'):
                stop_training = True
            if inp.startswith('SAVE'):
                fpath = inp.split(' ')[-1]
                stats['best_model'] = (best_val, best_model.cpu())
                torch.save({
                    'model': copy.deepcopy(model).cpu(),
                    'stats': stats,
                    'opt': copy.deepcopy(opt).cpu()
                }, fpath)
                PR(f'Saved to {fpath}')
            if inp.startswith('GAP'):
                _, gap = inp.split(' ')
                gap = int(gap)
                print ("epoch gap: {} -> {}".format(epoch_gap, gap))
                epoch_gap = gap
            if inp.startswith('MAXEPOCHS'):
                _, me = inp.split(' ')
                me = int(me)
                print ("max_epochs: {} -> {}".format(max_epochs, me))
                max_epochs = me
            
    stats['best_model'] = (best_val, best_model.cpu())
    stats['init_model'] = init_model
    return stats


def fit_model(model, loss, opt, train_dl, valid_dl, sch=None, epsilon=1e-2, is_loss_epsilon=False, update_gap=50, update_print_gap=50, gap=None, 
              print_info=True, save_grads=False, test_dl=None, skip_epoch_eval=True, sample_pct=0.5, sample_loss_threshold=0.75, save_models=False, 
              print_grads=False, print_model_layers=False, tr_batch_fn=None, te_batch_fn=None, device=None, max_updates=800_000, patience_updates=1, 
              enable_redo=False, save_best_model=True, save_init_model=True, max_epochs=100000, **misc):

    # setup update metadata
    MAX_LOSS_VAL = 1000000.
    PR = lambda x: print (x) if print_info else None
    use_epoch = False
    if gap is not None: update_gap = update_print_gap = gap
    bs_ratio = int(len(train_dl.dataset)/float(train_dl.batch_size))
    act_update_gap = update_gap if not use_epoch else update_gap*bs_ratio
    act_pr_update_gap = update_print_gap if not use_epoch else update_print_gap*bs_ratio
    PR("accuracy/loss measured every {} updates".format(act_update_gap))

    if save_models:
        PR("saving models every {} updates".format(act_update_gap))

    PR("update_print_gap: {}, epss: {}, bs: {}, device: {}".format(act_pr_update_gap, epsilon, train_dl.batch_size, device or 'cpu'))

    # init_save setup
    init_model = copy.deepcopy(model).cpu() if save_init_model else None

    # redo setup
    if enable_redo:
        init_model_sd = copy.deepcopy(model.state_dict())
        init_opt_sd = copy.deepcopy(opt.state_dict())
    else:
        init_model_sd = None
        init_opt_sd = None

    # best model setup
    best_val, best_model = 0, None

    # tracking setup
    start_time = time.time()
    num_evals, num_epochs, num_updates, num_patience = 0, 0, 0, 0
    stats = dict(loss_tr=[], loss_te=[], acc_tr=[], acc_te=[], acc_test=[], loss_test=[], models=[], gradients=[])
    if save_models: stats['models'].append(copy.deepcopy(model).cpu())
    first_run, converged = True, False
    print_stats_flag = update_print_gap is not None
    exceeded_max = False
    diverged = False

    def _evaluate(device=device):        
        model.eval()
        with torch.no_grad():
            prev_loss = stats['loss_tr'][-1] if stats['loss_tr'] else 1.
            tr_sample_pct = sample_pct if prev_loss > sample_loss_threshold else 1.
            acc_tr, loss_tr = compute_loss_and_accuracy_from_dl(train_dl,model,loss,sample_pct=tr_sample_pct,device=device,transform_fn=tr_batch_fn)
            acc_te, loss_te = compute_loss_and_accuracy_from_dl(valid_dl,model,loss,sample_pct=1.,device=device,transform_fn=te_batch_fn)
            acc_tr, loss_tr, acc_te, loss_te = map(lambda x: x.item(), [acc_tr, loss_tr, acc_te, loss_te])
            stats['loss_tr'].append(loss_tr)
            stats['loss_te'].append(loss_te)
            stats['acc_tr'].append(acc_tr)
            stats['acc_te'].append(acc_te)

            if test_dl is not None:
                acc_test, loss_test = compute_loss_and_accuracy_from_dl(test_dl,model,loss,sample_pct=1.,device=device,transform_fn=te_batch_fn)
                acc_test, loss_test = acc_test.item(), loss_test.item()
                stats['acc_test'].append(acc_test)
                stats['loss_test'].append(loss_test)

            if save_models:
                stats['models'].append(copy.deepcopy(model).cpu())

    def _update(x,y,diff_device, device=device, save_grads=False, print_grads=False):
        model.train()

        # if diff_device:
        #     x = x.to(device, non_blocking=False)
        #     y = y.to(device, non_blocking=False)

        opt.zero_grad()
        out = model(x)
        if loss is F.cross_entropy or loss is hinge_loss:
            bloss = loss(out, y)
        elif loss is F.binary_cross_entropy_with_logits:
            bloss = loss(out, y.float().unsqueeze(1))
        else:
            try:
                bloss = loss(out, y)
            except:
                assert False, "unknown loss function"

        bloss.backward()
        if print_grads and print_info: print_model_gradients(model)
        #clip_gradient(model, clip_value)
        opt.step()

        if save_grads:
            g = {k: v.grad.data.cpu().numpy() for k, v in model.named_parameters() if v.requires_grad}
            stats['gradients'].append(g)

        opt.zero_grad()
        model.eval()

    def print_time():
        end_time = time.time()
        minutes, seconds = divmod(end_time-start_time, 60)
        gap_valid = len(stats['acc_tr']) > 0
        gap = round(stats['acc_tr'][-1]-stats['acc_te'][-1],4) if gap_valid else 'na'
        PR("converged after {} epochs in {}m {:1f}s, gap: {}".format(num_epochs, minutes, seconds, gap))

    def print_stats(force_print=False):

        if test_dl is None:
            atr, ate, ltr = [stats[k][-1] for k in ['acc_tr', 'acc_te', 'loss_tr']]
            PR("{} {:.4f} {:.4f} {:.4f}".format(num_updates, atr, ate, ltr))
            if not print_info and force_print:
                print ("{} {:.4f} {:.4f} {:.4f}".format(num_updates, atr, ate, ltr))
        else:
            atr, aval, ate, ltr = [stats[k][-1] for k in ['acc_tr', 'acc_te', 'acc_test', 'loss_tr']]
            PR("{} {:.4f} {:.4f} {:.4f} {:.4f}".format(num_updates, atr, aval, ate, ltr))
            if not print_info and force_print:
                print ("{} {:.4f} {:.4f} {:.4f} {:.4f}".format(num_updates, atr, aval, ate, ltr))

    #xb_, yb_ = next(iter(train_dl))
    diff_device = True #xb_.device != device

    if test_dl is None: PR("#updates, train acc, test acc, train loss")
    else: PR("#updates, train acc, val acc, test acc, train loss")

    while not converged or num_patience < patience_updates:
        try:
            model.train()
            for xb, yb in train_dl:

                if tr_batch_fn:
                    xb, yb = tr_batch_fn(xb, yb)

                if diff_device:
                    xb = xb.to(device, non_blocking=False)
                    yb = yb.to(device, non_blocking=False)

                if converged:
                    num_patience += 1

                if converged and num_patience == patience_updates:
                    _evaluate()
                    print_stats()
                    break

                # update flag for printing gradients
                update_flag = print_model_layers and (num_updates == 0 or (num_updates % act_update_gap == 0 and print_grads))
                _update(xb, yb, diff_device, device=device, save_grads=save_grads, print_grads=update_flag)

                if (num_evals == 0 or num_updates % act_update_gap == 0):
                    num_evals += 1
                    _evaluate()
                    print_stats()

                    val_acc = stats['acc_te'][-1]
                    if num_updates > 0 and val_acc >= best_val:
                        best_val = val_acc
                        best_model = copy.deepcopy(model).eval()

                    # check if loss has diverged
                    loss_val = max(stats['loss_tr'][-1], stats['loss_te'][-1])
                    if loss_val > MAX_LOSS_VAL: diverged = True
                    if not np.isfinite(loss_val): diverged = True


                    if is_loss_epsilon: stop = stats['loss_tr'][-1] < epsilon
                    else: stop = stats['acc_tr'][-1] >= 1-epsilon

                    if not converged and diverged:
                        converged = True
                        print_time()
                        PR("loss diverging...exiting".format(patience_updates))

                    if not converged and stop:
                        converged = True
                        print_time()
                        PR("init-ing patience ({} updates)".format(patience_updates))

                num_updates += 1
                first_run = False

                if num_updates > max_updates:
                    converged = True
                    exceeded_max = True
                    num_patience = patience_updates
                    PR("Exceeded max updates")
                    print_stats()
                    print_time()
                    break

            # re-eval at the end of epoch
            if not converged:
                num_epochs += 1
            
            if not converged and num_epochs >= max_epochs:
                converged = True
                exceeded_max = True
                num_patience = patience_updates
                PR("Exceeded max epochs")
                print_stats()
                print_time()
                break

            if not skip_epoch_eval:
                _evaluate()
                print_stats()

                if is_loss_epsilon: stop = stats['loss_tr'][-1] < epsilon
                else: stop = stats['acc_tr'][-1] >= 1-epsilon

                if not converged and stop:
                    converged = True
                    print_time()
                    PR("init-ing patience ({} updates)".format(patience_updates))

                if num_patience >= patience_updates:
                    _evaluate()
                    print_stats()
                    break

            # update LR via scheduler
            if sch is not None:
                cur_lr = next(iter(opt.param_groups))['lr']
                sch.step()
                new_lr = next(iter(opt.param_groups))['lr']
                if new_lr != cur_lr:
                    PR('Epoch {}, LR : {} -> {}'.format(num_epochs, cur_lr, new_lr))

        except KeyboardInterrupt:
            inp = input("LR num or Q or GAP num or SAVE fpath or EVAL or REDO: ")
            if inp.startswith('LR'):
                lr = float(inp.split(' ')[-1])
                cur_lr = next(iter(opt.param_groups))['lr']
                PR("LR: {} - > {}".format(cur_lr, lr))
                for g in opt.param_groups: g['lr'] = lr
            elif inp.startswith('GAP'):
                gap = int(inp.split(' ')[-1])
                act_update_gap = act_pr_update_gap = gap
            elif inp == "Q":
                converged = True
                num_patience = patience_updates
                print_time()
            elif inp.startswith('SAVE'):
                fpath = inp.split(' ')[-1]
                torch.save({
                    'model': model,
                    'opt': opt,
                    'update_gap': update_gap
                }, fpath)
            elif inp == 'EVAL':
                _evaluate()
                print_stats(True)
            elif inp == 'REDO':
                if enable_redo:
                    model.load_state_dict(init_model_sd)
                    opt.load_state_dict(init_opt_sd)
                else:
                    print ("REDO disabled")

    best_test = None
    if test_dl is not None:
        best_test = compute_loss_and_accuracy_from_dl(test_dl, best_model, loss, sample_pct=1.0, device=device)[0].item()

    stats['num_updates'] = num_updates
    stats['num_epochs'] = num_epochs
    stats['update_gap'] = update_gap

    stats['best_model'] = (best_val, best_test, best_model.cpu() if best_model else model.cpu())
    stats['init_model'] = init_model
    if save_models: stats['models'].append(copy.deepcopy(model).cpu())

    stats['x_updates']= list(range(0, num_evals*(update_gap+1), update_gap))
    stats['x'] = stats['x_updates'][:]
    stats['x_epochs'] = list(range(num_epochs))
    stats['gap'] = stats['acc_tr'][-1]-stats['acc_te'][-1]
    return stats

def save_pickle(fname, d, mode='w'):
    with open(fname, mode) as f:
        pickle.dump(d, f)

def load_pickle(fname, mode='r'):
    with open(fname, mode) as f:
        return pickle.load(f)

def update_ax(ax, title=None, xlabel=None, ylabel=None, legend_loc='best', ticks=True, ticks_fs=10, label_fs=12, legend_fs=12, title_fs=14, hide_xlabels=False, hide_ylabels=False, despine=True):
    if title: ax.set_title(title, fontsize=title_fs)
    if xlabel: ax.set_xlabel(xlabel, fontsize=label_fs)
    if ylabel: ax.set_ylabel(ylabel, fontsize=label_fs)
    if legend_loc: ax.legend(loc=legend_loc, fontsize=legend_fs)
    if despine: sns.despine(ax=ax)

    if ticks:
        # ax.minorticks_on()
        ax.tick_params(direction='in', length=6, width=2, colors='k', which='major', top=False, right=False)
        ax.tick_params(direction='in', length=4, width=1, colors='k', which='minor', top=False, right=False)
        ax.tick_params(labelsize=ticks_fs)

    if hide_xlabels: ax.set_xticks([])
    if hide_ylabels: ax.set_yticks([])
    return ax
