import sys
import numpy as np
import argparse
import copy
import random
import json

import torch
from torch.autograd import grad
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import torch.utils.data as data_utils

from .base_eval import BaseEval
from utils.match_function import get_matched_pairs

class PerDomainAcc(BaseEval):
    
    def __init__(self, args, train_dataset, val_dataset, test_dataset, base_res_dir, run, cuda):
        super().__init__(args, train_dataset, val_dataset, test_dataset, base_res_dir, run, cuda)
                
    def get_metric_eval(self):
        
        case= self.args.acc_data_case        
        if case=='train':
            dataset= self.train_dataset['data_loader']
            total_domains= self.train_dataset['total_domains']
            domain_list= self.train_dataset['domain_list']
            base_domain_size= self.train_dataset['base_domain_size']
            domain_size_list= self.train_dataset['domain_size_list']

        elif case== 'val':
            dataset= self.val_dataset['data_loader']
            total_domains= self.val_dataset['total_domains']
            domain_list= self.val_dataset['domain_list']
            base_domain_size= self.val_dataset['base_domain_size']
            domain_size_list= self.val_dataset['domain_size_list']

        elif case== 'test':
            dataset= self.test_dataset['data_loader']
            total_domains= self.test_dataset['total_domains']
            domain_list= self.test_dataset['domain_list']
            base_domain_size= self.test_dataset['base_domain_size']
            domain_size_list= self.test_dataset['domain_size_list']
        
        acc_per_domain={}
        size_per_domain={}        
        for batch_idx, (x_e, y_e ,d_e, idx_e, obj_e) in enumerate(dataset):
            with torch.no_grad():
                x_e= x_e.to(self.cuda)
                y_e= torch.argmax(y_e, dim=1).to(self.cuda)
                d_e= torch.argmax(d_e, dim=1)
                
                #Forward Pass
                out= self.forward(x_e)                
                loss_e= torch.mean(F.cross_entropy(out, y_e.long()).to(self.cuda))
                
                for domain in np.unique(d_e.cpu().numpy()):
                    indices= d_e==domain
                    y_c= y_e[indices]
                    out_c= out[indices]

                    if y_c.shape[0]:
                        acc= torch.sum( torch.argmax(out_c, dim=1) == y_c ).item()
                        size= y_c.shape[0]

                        if domain in acc_per_domain:
                            acc_per_domain[domain] += acc
                            size_per_domain[domain] += size
                        else:
                            acc_per_domain[domain] = acc
                            size_per_domain[domain] = size                            
                
        for domain in acc_per_domain.keys():
            acc_per_domain[domain]= 100*acc_per_domain[domain]/size_per_domain[domain]
            print( 'Per Domain Acc: ', str(domain), acc_per_domain[domain], end =" " )
            self.metric_score[domain]= acc_per_domain[domain] 

        return 