import sys
import random
import os, copy, pickle, time
import argparse
import itertools
from collections import defaultdict, Counter, OrderedDict
import matplotlib.pyplot as plt

import numpy as np
import seaborn as sns
import torch
import torchvision
from torch import optim, nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
import pandas as pd

import utils.scripts.gpu_utils as gu
import utils.scripts.gendata as gendata

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

DEVICE_ID = 0 # GPU_ID or None (CPU)
DEVICE = gu.get_device(DEVICE_ID)

def get_data(num_samples, spur_corr, slab_noise, total_slabs, data_case, seed, mask_linear):
    
#     if total_slabs== 5:
#         slab_5_flag= 1
#         slab_noise_5= slab_noise
#         slab_7_flag= 0
#         slab_noise_7= 0.0
#     elif total_slabs== 7:
#         slab_5_flag= 0
#         slab_noise_5= 0.0
#         slab_7_flag= 1 
#         slab_noise_7= slab_noise
    
#     print('Noise in Slab Feature: ', slab_noise)
#     c = config =  {
#         'num_train': num_samples, # training dataset size
#         'dim': 2, # input dimension
#         'lin_margin': 0.1, # linear margin
#         'slab_margin': 0.1, # slab margin,
#         'same_margin': True, # keep same margin
#         'random_transform': False, # apply random (orthogonal) transformation
#         'width': 1, # data width in standard basis
#         'num_lin': 1, # number of linear components
#         'num_slabs': slab_5_flag, #. number of 5 slabs
#         'num_slabs7': slab_7_flag, # number of 7 slabs
#         'num_slabs3': 0, # number of 3 slabs
#         'bs': 256, # batch size
#         'corrupt_lin': spur_corr, # p_noise
#         'corrupt_lin_margin': True, # noise model
#         'corrupt_slab': slab_noise_5, # slab corruption
#         'corrupt_slab7': slab_noise_7, # slab corruption
#         'num_test': 0, # test dataset size
#         'hdim': 100, # model width
#         'hl': 2, # model depth
#         'mtype': 'fcn', # model architecture
#         'device': gu.get_device(DEVICE_ID), # GPU device
#         'lr': 0.1, # step size
#         'weight_decay': 5e-5 # weight decay
#     }

#     smargin = c['lin_margin'] if c['same_margin'] else c['slab_margin']
#     data_func = gendata.generate_ub_linslab_data_v2
#     spc = [3]*c['num_slabs3']+[5]*c['num_slabs'] + [7]*c['num_slabs7']
#     data = data_func(c['num_train'], c['dim'], c['lin_margin'], slabs_per_coord=spc, eff_slab_margin=smargin, random_transform=c['random_transform'], N_te=c['num_test'],
#                      corrupt_lin_margin=c['corrupt_lin_margin'], num_lin=c['num_lin'], num_slabs=c['num_slabs3']+c['num_slabs']+c['num_slabs7'], width=c['width'], bs=c['bs'], corrupt_lin=c['corrupt_lin'], corrupt_slab=c['corrupt_slab'], corrupt_slab7=c['corrupt_slab7'])            
    
    #Load data from json file
    base_dir= '/home/t-dimaha/RobustDG/robustdg/data/datasets/slab/'
    data_dir= base_dir + 'total_slabs_' + str(total_slabs) + '_slab_noise_' + str(slab_noise) + '_spur_corr_' + str(spur_corr) + '_case_' + str(data_case) + '_seed_' + str(seed)
    
    with open(data_dir + '.pickle', 'rb') as handle:
        data = pickle.load(handle)

        
#     Mask model requirement
    if mask_linear:
        print('Linear Feature Masked')
        data['X'][:, 0]= 0.0 + 0*data['X'][:, 0]    
    
    #Project data 
    W = data['W']
    X, Y = data['X'][:num_samples], data['Y'][:num_samples]
    #X = X.numpy().dot(W.T)
    X = X.numpy()
    Y = Y.numpy()
    print('Seed: ', X[0])
    
    #Generate objects/slab_id array 
    # Equation: k*slab_len + (k-1)*0.2 = 2: k is total number of slabs
    # Slabs: [-1, -1 + slab_len], [-1+slab_len+0.2,  -1 + 0.2 + 2*slab_len ]
    # Slabs:  For i in range(0, k-1): [-1 + i*(0.2 + slab_len), -1 + slab_len + i*(0.2 + slab_len) ]
    
    k=total_slabs
    slab_len= (2 - 0.2*(k-1))/k    
    O=np.zeros(X.shape[0])
    for idx in range(X.shape[0]):
        for slab_id in range(k):
            start= -1 + slab_id*(0.2 + slab_len)
            end= start+ slab_len    
            if X[idx, 1] >= start and X[idx, 1] <= end:
                O[idx]= slab_id
                break
            
    return data, X, Y, O

