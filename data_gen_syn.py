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
import pickle

# sys.path.insert(0, '/../utils/')
import os
cwd = os.getcwd()
print(cwd)

import utils.scripts.gpu_utils as gu
import utils.scripts.gendata as gendata

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

DEVICE_ID = 0 # GPU_ID or None (CPU)
DEVICE = gu.get_device(DEVICE_ID)

#Generate Dataset for Slab Dataset
base_dir= 'data/datasets/slab/'
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

num_samples= 10000
total_slabs= 7
slab_noise_list= [0.0, 0.10]
spur_corr_list= [0.0, 0.10, 0.20, 0.90]

for seed in range(10):
    np.random.seed(seed*10)                     
    for slab_noise in slab_noise_list:
        for spur_corr in spur_corr_list: 
            for case in ['train', 'val', 'test']:

                data_dir= base_dir + 'total_slabs_' + str(total_slabs) + '_slab_noise_' + str(slab_noise) + '_spur_corr_' + str(spur_corr) + '_case_' + str(case) + '_seed_' + str(seed)
            
                if total_slabs== 5:
                    slab_5_flag= 1
                    slab_noise_5= slab_noise
                    slab_7_flag= 0
                    slab_noise_7= 0.0
                elif total_slabs== 7:
                    slab_5_flag= 0
                    slab_noise_5= 0.0
                    slab_7_flag= 1 
                    slab_noise_7= slab_noise

                print('Noise in Slab Feature: ', slab_noise)
                c = config =  {
                    'num_train': num_samples, # training dataset size
                    'dim': 2, # input dimension
                    'lin_margin': 0.1, # linear margin
                    'slab_margin': 0.1, # slab margin,
                    'same_margin': True, # keep same margin
                    'random_transform': False, # apply random (orthogonal) transformation
                    'width': 1, # data width in standard basis
                    'num_lin': 1, # number of linear components
                    'num_slabs': slab_5_flag, #. number of 5 slabs
                    'num_slabs7': slab_7_flag, # number of 7 slabs
                    'num_slabs3': 0, # number of 3 slabs
                    'bs': 256, # batch size
                    'corrupt_lin': spur_corr, # p_noise
                    'corrupt_lin_margin': True, # noise model
                    'corrupt_slab': slab_noise_5, # slab corruption
                    'corrupt_slab7': slab_noise_7, # slab corruption
                    'num_test': 0, # test dataset size
                    'hdim': 100, # model width
                    'hl': 2, # model depth
                    'mtype': 'fcn', # model architecture
                    'device': gu.get_device(DEVICE_ID), # GPU device
                    'lr': 0.1, # step size
                    'weight_decay': 5e-5 # weight decay
                }

                smargin = c['lin_margin'] if c['same_margin'] else c['slab_margin']
                data_func = gendata.generate_ub_linslab_data_v2
                spc = [3]*c['num_slabs3']+[5]*c['num_slabs'] + [7]*c['num_slabs7']
                data = data_func(c['num_train'], c['dim'], c['lin_margin'], slabs_per_coord=spc, eff_slab_margin=smargin, random_transform=c['random_transform'], N_te=c['num_test'],
                                 corrupt_lin_margin=c['corrupt_lin_margin'], num_lin=c['num_lin'], num_slabs=c['num_slabs3']+c['num_slabs']+c['num_slabs7'], width=c['width'], bs=c['bs'], corrupt_lin=c['corrupt_lin'], corrupt_slab=c['corrupt_slab'], corrupt_slab7=c['corrupt_slab7'])    
                
                print(data_dir)                    
                with open(data_dir + '.pickle', 'wb') as fname:
                    pickle.dump(data, fname, protocol=pickle.HIGHEST_PROTOCOL)
