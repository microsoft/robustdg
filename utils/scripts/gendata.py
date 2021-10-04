import numpy as np
import scipy.stats as scs
import random
from collections import Counter
import torch
from torch.utils.data import TensorDataset, DataLoader
import utils.scripts.utils as utils
import utils.scripts.gpu_utils as gu

def _prep_data(X, Y, N_tr, N_te, bs, nw, pm, w, orth_matrix=None):
    X_te, Y_te = torch.Tensor(X[:N_te,:]), torch.Tensor(Y[:N_te])
    X_tr, Y_tr = torch.Tensor(X[N_te:,:]), torch.Tensor(Y[N_te:])
    Y_te, Y_tr = map(lambda Z: Z.long(), [Y_te, Y_tr])
    
    tr_dl = DataLoader(TensorDataset(X_tr, Y_tr), batch_size=bs, num_workers=nw, pin_memory=pm, shuffle=True)
#     te_dl = DataLoader(TensorDataset(X_te, Y_te), batch_size=bs, num_workers=nw, pin_memory=pm, shuffle=False)
    te_dl = DataLoader(TensorDataset(X_tr, Y_tr), batch_size=bs, num_workers=nw, pin_memory=pm, shuffle=False)

    return {
        'X': torch.tensor(X).float(),
        'Y': torch.tensor(Y).long(),
        'w': w,
        'tr_dl': tr_dl,
        'te_dl': te_dl,
        'N': (N_tr, N_te),
        'W': orth_matrix
    }

def _get_random_data(N, dim, scale):
    X = np.random.uniform(size=(N, dim))
    X *= scale
    Y = np.random.choice([0,1], size=N)
    return X, Y

def generate_linsep_data_v2(N_tr, dim, eff_margin, width=10., bs=256, scale_noise=True, pm=True, nw=0, no_width=False, N_te=5000): # no unif_max.
    assert eff_margin < 1, "equal range constraint"
    margin = eff_margin if no_width else eff_margin*width

    N = N_tr + N_te
    w = np.zeros(shape=dim)
    w[0] = 1

    X, Y = _get_random_data(N, dim, width if scale_noise else 1.)

    U = np.random.uniform(size=N)
    if no_width: X[:,0] = (2*Y-1)*margin
    else: X[:, 0] = (2*Y-1)*(margin + (width-margin)*U)

    P = np.random.permutation(X.shape[0])
    X, Y = X[P,:], Y[P]

    return _prep_data(X, Y, N_tr, N_te, bs, nw, pm, w)

def sample_from_unif_union_of_unifs(unifs, size):
    x = []
    choices = Counter(np.random.choice(list(range(len(unifs))), size=size))
    for choice, sz in choices.items():
        s = np.random.uniform(low=unifs[choice][0], high=unifs[choice][1], size=sz)
        x.append(s)
    x = np.concatenate(x)
    return x

def generate_ub_linslab_data_diffmargin_v2(N_tr, dim, eff_lin_margins, eff_slab_margins,
                                           slabs_per_coord, slab_p_vals, corrupt_lin=0., corrupt_slab=0.,
                                           corrupt_slab7=0., scale_noise=True, width=10., lin_coord=0, lin_shift=0.,
                                           slab_shift=0., indep_slabs=True, bs=256, pm=True, nw=0, N_te=10000,
                                           random_transform=False, corrupt_lin_margin=False, corrupt_5slab_margin=False):
    get_unif = lambda a: np.random.uniform(size=a)
    get_bool = lambda a: np.random.choice([0,1], size=a)
    get_sign = lambda a: 2*get_bool(a)-1.

    def get_slab_width(NS, B, SM):
        if NS==3: return (2.*B-4.*SM)/3.
        if NS==5: return (2.*B-8.*SM)/5.
        if NS==7: return (2.*B-12.*SM)/7.
        return None

    num_lin, num_slabs = map(len, [eff_lin_margins, eff_slab_margins])
    assert 0 <= corrupt_lin <= 1, "input is probability"
    assert num_lin + num_slabs <= dim, "dim constraint, num_lin: {}, num_slabs: {}, dim: {}".format(num_lin, num_slabs, dim)
    for elm in eff_lin_margins: assert 0 < elm < 1, "equal range constraint (0 < eff_lin_margin={} < 1)".format(elm)
    for esm in eff_slab_margins: assert 0 < esm < 1, "equal range constraint (0 < eff_slab_margin={} < 0.25)".format(esm)

    lin_margins = list(map(lambda x: x*width, eff_lin_margins))
    slab_margins = list(map(lambda x: x*width, eff_slab_margins))

    # hyperplane
    N = N_tr + N_te
    half_N = N//2
    w = np.zeros(shape=dim); w[0] = 1

    X, Y = _get_random_data(N, dim, width if scale_noise else 1.)
    nrange = list(range(N))
    # linear
    total_corrupt = int(round(N*corrupt_lin))
    no_linear = num_lin == 0
    if not no_linear:
        for coord, lin_margin in enumerate(lin_margins):
            if indep_slabs:
                P = np.random.permutation(N)
                X, Y = X[P, :], Y[P]
            X[:, coord] = (2*Y-1)*(lin_margin+(width-lin_margin)*get_unif(N)) + lin_shift*width

            # corrupt linear coordinate
            if total_corrupt > 0:
                corrupt_sample = np.random.choice(nrange, size=total_corrupt, replace=False)
                if corrupt_lin_margin:
                    X[corrupt_sample, 0] = np.random.uniform(low=-lin_margin, high=lin_margin, size=total_corrupt)
                else:
                    X[corrupt_sample, 0] *= -1

    # slabs
    i = (num_lin)*int(not no_linear)
    for idx, coord in enumerate(range(i, i+num_slabs)):
        slab_per = slabs_per_coord[idx]
        assert slab_per in [3, 5, 7], "Invalid slabs_per_coord"

        slab_pval = slab_p_vals[idx]
        slab_margin = slab_margins[idx]
        slab_width = get_slab_width(slab_per, width, slab_margin)

        if indep_slabs:
            P = np.random.permutation(N)
            X, Y = X[P, :], Y[P]
        
        print('slab_per', slab_per)
        
        if slab_per == 3:
            # positive slabs
            idx_p = (Y==1).nonzero()[0]
            offset = 0.5*slab_width + 2*slab_margin
            X[idx_p, coord] = get_sign(len(idx_p))*(offset+slab_width*get_unif(len(idx_p)))

            # negative center
            idx_n = (Y==0).nonzero()[0]
            X[idx_n, coord] = 0.5*get_sign(len(idx_n))*slab_width*get_unif(len(idx_n))

        if slab_per == 5:
            # positive slabs
            idx_p = (Y==1).nonzero()[0]
            offset = (width+6*slab_margin)/5.
            X[idx_p, coord] = get_sign(len(idx_p))*(offset+slab_width*get_unif(len(idx_p)))

            # negative slabs partitioned using p val
            idx_n = (Y==0).nonzero()[0]
            in_ctr = np.random.choice([0,1], p=[1-slab_pval, slab_pval], size=len(idx_n))
            idx_nc, idx_ns = idx_n[(in_ctr==1)], idx_n[(in_ctr==0)]

            # negative center
            X[idx_nc, coord] = 0.5*get_sign(len(idx_nc))*slab_width*get_unif(len(idx_nc))

            # negative sides
            offset = (8*slab_margin+3*width)/5.
            X[idx_ns, coord] = get_sign(len(idx_ns))*(offset+slab_width*get_unif(len(idx_ns)))

            # corrupt slab 5
            print('Gen Data Slab Corruption: ', round(N*corrupt_slab))
            total_corrupt = int(round(N*corrupt_slab))
            if total_corrupt > 0:
                print('Yes, slab corruption is being applied')
                if corrupt_5slab_margin:
                    offset1 = (width+6*slab_margin)/5.
                    offset2 = (8*slab_margin+3*width)/5.
                    unifs = [
                        (0.5*slab_width, offset1),
                        (offset1+slab_width, offset2),
                        (-offset1, -0.5*slab_width),
                        (-offset2, -(offset1+slab_width))
                    ]

                    idx = np.random.choice(range(N), size=total_corrupt, replace=False)
                    X[idx, coord] = sample_from_unif_union_of_unifs(unifs, total_corrupt)
                else:
                    # get corrupt sample
                    idx = np.random.choice(range(N), size=total_corrupt, replace=False)
                    idx_p = idx[np.argwhere((Y[idx]==1))].reshape(-1)
                    idx_n = idx[np.argwhere((Y[idx]==0))].reshape(-1)

                    # move negative points to random positive slabs
                    offset = (0.5*slab_width+2*slab_margin)
                    X[idx_n, coord] = torch.Tensor(get_sign(len(idx_n))*(offset+slab_width*get_unif(len(idx_n))))

                    # pick negative slab for each positve point
                    mv_to_ctr = np.random.choice([0, 1], size=len(idx_p))
                    idx_p_ctr = idx_p[np.argwhere(mv_to_ctr==1)].reshape(-1)
                    idx_p_sid = idx_p[np.argwhere(mv_to_ctr==0)].reshape(-1)

                    # move positive points to negative slabs
                    X[idx_p_ctr, coord] = torch.Tensor(0.5*get_sign(len(idx_p_ctr))*slab_width*get_unif(len(idx_p_ctr)))

                    # move negative points to positve slabs
                    offset = 1.5*slab_width + 4*slab_margin
                    X[idx_p_sid, coord] = torch.Tensor(get_sign(len(idx_p_sid))*(offset+slab_width*get_unif(len(idx_p_sid))))

        if slab_per == 7:
            # positive slabs
            idx_p = (Y==1).nonzero()[0]
            in_s0 = np.random.choice([0,1], p=[1-slab_pval, slab_pval], size=len(idx_p))
            idx_p0, idx_p1 = idx_p[(in_s0==1)], idx_p[(in_s0==0)]

            # positive slab 0 (inner)
            offset = 0.5*slab_width+2*slab_margin
            X[idx_p0, coord] = get_sign(len(idx_p0))*(offset+slab_width*get_unif(len(idx_p0)))

            # positive slab 1 (outer)
            offset = 2.5*slab_width+6*slab_margin
            X[idx_p1, coord] = get_sign(len(idx_p1))*(offset+slab_width*get_unif(len(idx_p1)))

            # negative slabs
            idx_n = (Y==0).nonzero()[0]
            in_s0 = get_bool(len(idx_n))
            idx_n0, idx_n1 = idx_n[(in_s0==1)], idx_n[(in_s0==0)]

            # negative slab 0 (center)
            X[idx_n0, coord] = 0.5*get_sign(len(idx_n0))*slab_width*get_unif(len(idx_n0))

            # negative slab 1 (outer)
            offset = 1.5*slab_width+4*slab_margin
            X[idx_n1, coord] = get_sign(len(idx_n1))*(offset+slab_width*get_unif(len(idx_n1)))

            # corrupt slab7
            total_corrupt = int(round(N*corrupt_slab7))
            if total_corrupt > 0:
                print('Yes, slab corruption is being applied')
                # corrupt data
                idx = np.random.choice(range(len(X)), size=total_corrupt, replace=False)
                idx_p = idx[np.argwhere((Y[idx]==1))].reshape(-1)
                idx_n = idx[np.argwhere((Y[idx]==0))].reshape(-1)

                # pick positive slab for each negative slab
                mv_to_inner = get_bool(len(idx_n))
                idx_n_inner = idx_n[np.argwhere(mv_to_inner==1)].reshape(-1)
                idx_n_outer = idx_n[np.argwhere(mv_to_inner==0)].reshape(-1)

                # move to idx_n_inner and outer
                offset = 0.5*slab_width+2*slab_margin
                X[idx_n_inner, coord] = torch.Tensor(get_sign(len(idx_n_inner))*(offset+slab_width*get_unif(len(idx_n_inner))))
                offset = 2.5*slab_width+6*slab_margin
                X[idx_n_outer, coord] = torch.Tensor(get_sign(len(idx_n_outer))*(offset+slab_width*get_unif(len(idx_n_outer))))

                # pick negative slab for each positive point
                mv_to_ctr = get_bool(len(idx_p))
                idx_p_ctr = idx_p[np.argwhere(mv_to_ctr==1)].reshape(-1)
                idx_p_sid = idx_p[np.argwhere(mv_to_ctr==0)].reshape(-1)

                # move to idx_n_inner and outer
                X[idx_p_ctr, coord] = torch.Tensor(0.5*get_sign(len(idx_p_ctr))*(slab_width*get_unif(len(idx_p_ctr))))
                offset = 1.5*slab_width+4*slab_margin
                X[idx_p_sid, coord] = torch.Tensor(get_sign(len(idx_p_sid))*(offset+slab_width*get_unif(len(idx_p_sid))))

        # shift
        X[:, coord] += slab_shift*width

    # reshuffle
    P = np.random.permutation(N)
    X, Y = X[P,:], Y[P]

    # lin coord position
    if not random_transform and lin_coord != 0:
        X[:, [0, lin_coord]] = X[:, [lin_coord, 0]]

    # transform
    W = np.eye(dim)
    if random_transform: W = utils.get_orthonormal_matrix(dim)
    X  = X.dot(W)

    return _prep_data(X, Y, N_tr, N_te, bs, nw, pm, w, orth_matrix=W)


def generate_ub_linslab_data_v2(N_tr, dim, eff_lin_margin, eff_slab_margin=None, lin_coord=0,
                                corrupt_lin=0., corrupt_slab=0., corrupt_slab3=0., corrupt_slab7=0.,
                                scale_noise=True, num_lin=1, lin_shift=0., slab_shift=0., random_transform=False,
                                num_slabs=1, slabs_per_coord=5, width=10., indep_slabs=True, no_linear=False,
                                bs=256, pm=True, nw=0, N_te=10000, corrupt_lin_margin=False, slab5_pval=3/4.,
                                slab3_pval=1/2., slab7_pval=7/8., corrupt_5slab_margin=False):
    slab_p_map = {5: slab5_pval, 7: slab7_pval, 3: slab3_pval}
    slabs_per_coord = [slabs_per_coord]*num_slabs if type(slabs_per_coord) is int else slabs_per_coord[:]
    for x in slabs_per_coord: assert x in slab_p_map
    slab_p_vals = [slab_p_map[x] for x in slabs_per_coord]
    lms = [eff_lin_margin]*num_lin
    sms = eff_slab_margin if type(eff_slab_margin) is list else [eff_slab_margin]*num_slabs
    return generate_ub_linslab_data_diffmargin_v2(N_tr, dim, lms, sms, slabs_per_coord, slab_p_vals, lin_coord=lin_coord, corrupt_slab=corrupt_slab,
                                                  corrupt_slab7=corrupt_slab7, corrupt_lin=corrupt_lin, scale_noise=scale_noise, width=width,
                                                  lin_shift=lin_shift, slab_shift=slab_shift, random_transform=random_transform, indep_slabs=indep_slabs,
                                                  pm=pm, bs=bs, corrupt_lin_margin=corrupt_lin_margin, nw=nw, N_te=N_te, corrupt_5slab_margin=corrupt_5slab_margin)


def get_lms_data(**kw):
    
    c = config =  {
        'num_train': 100_000,
        'dim': 20,
        'lin_margin': 0.1,
        'slab_margin': 0.1,
        'same_margin': False,
        'random_transform': False,
        'width': 1, # data width
        'bs': 256,
        'corrupt_lin': 0.0,
        'corrupt_lin_margin': False,
        'corrupt_slab': 0.0,
        'num_test': 2_000,
        'hdim': 200, # model width
        'hl': 2, # model depth
        'device': gu.get_device(0),
        'input_dropout': 0,
        'num_lin': 1,
        'num_slabs': 19,
        'num_slabs7': 0,
        'num_slabs3': 0,
    }
    
    c.update(kw)

    smargin = c['lin_margin'] if c['same_margin'] else c['slab_margin']
    data_func = generate_ub_linslab_data_v2
    spc = [3]*c['num_slabs3']+[5]*c['num_slabs'] + [7]*c['num_slabs7']
    data = data_func(c['num_train'], c['dim'], c['lin_margin'], slabs_per_coord=spc, eff_slab_margin=smargin, random_transform=c['random_transform'], N_te=c['num_test'],
                     corrupt_lin_margin=c['corrupt_lin_margin'], num_lin=c['num_lin'], num_slabs=c['num_slabs3']+c['num_slabs']+c['num_slabs7'], width=c['width'], bs=c['bs'], 
                     corrupt_lin=c['corrupt_lin'], corrupt_slab=c['corrupt_slab'])
    return data, c

