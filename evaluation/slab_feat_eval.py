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
from utils.match_function import get_matched_pairs, perfect_match_score
from utils.helper import l1_dist, l2_dist, embedding_dist, cosine_similarity


def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt

class SlabFeatEval(BaseEval):
    
    def __init__(self, args, train_dataset, val_dataset, test_dataset, base_res_dir, run, cuda):
        super().__init__(args, train_dataset, val_dataset, test_dataset, base_res_dir, run, cuda)
                
    def get_metric_eval(self):
        
        if self.args.match_func_data_case=='train':
            dataset= self.train_dataset['data_loader']
            total_domains= self.train_dataset['total_domains']
            domain_list= self.train_dataset['domain_list']
            base_domain_size= self.train_dataset['base_domain_size']
            domain_size_list= self.train_dataset['domain_size_list']
            
        elif self.args.match_func_data_case== 'val':
            dataset= self.val_dataset['data_loader']
            total_domains= self.val_dataset['total_domains']
            domain_list= self.val_dataset['domain_list']
            base_domain_size= self.val_dataset['base_domain_size']
            domain_size_list= self.val_dataset['domain_size_list']
        
        elif self.args.match_func_data_case== 'test':
            dataset= self.test_dataset['data_loader']
            total_domains= self.test_dataset['total_domains']
            domain_list= self.test_dataset['domain_list']
            base_domain_size= self.test_dataset['base_domain_size']
            domain_size_list= self.test_dataset['domain_size_list']
    
        pos_metric= 'cos'
        with torch.no_grad():
            
            penalty_ws=0    
            batch_size_counter=0
            #Batch iteration over single epoch
            for batch_idx, (x_e, y_e ,d_e, idx_e) in enumerate(dataset):
                
                x_e= x_e.to(self.cuda)
                y_e= torch.argmax(y_e, dim=1).to(self.cuda)                
                #Forward Pass
                out= self.phi(x_e)

                match_objs= np.unique(idx_e)
                feat= self.phi.feat_net(x_e)
                for obj in match_objs:
                    indices= idx_e == obj
                    feat_obj= feat[indices]
                    d_obj= d_e[indices]

                    match_domains= torch.unique(d_obj)

                    if len(match_domains) != len(torch.unique(d_e)):
        #                 print('Error: Positivty Violation, objects not present in all the domains')
                        continue

                    for d_i in range(len(match_domains)):
                        for d_j in range(len(match_domains)):
                            if d_j <= d_i:
                                continue
                            x1= feat_obj[ d_obj == d_i ]
                            x2= feat_obj[ d_obj == d_j ]
            
                            if pos_metric == 'l2':
                                x1= x1.view(x1.shape[0], 1, x1.shape[1])
                                penalty_ws+= float( torch.sum( torch.sum( torch.sum( (x1 -x2)**2, dim=2), dim=1 )) )
                            elif pos_metric == 'l1':
                                x1= x1.view(x1.shape[0], 1, x1.shape[1])
                                penalty_ws+= float( torch.sum( torch.sum( torch.sum( torch.abs(x1 -x2), dim=2), dim=1 )) )
                            elif pos_metric == 'cos':
                                penalty_ws+= float( torch.sum( torch.sum( sim_matrix(x1, x2), dim=1)) )
            
                            batch_size_counter+= x1.shape[0]*x2.shape[0]

                torch.cuda.empty_cache()    
                
        self.metric_score['Perfect Match Distance']= penalty_ws/batch_size_counter        
        print('Perfect Match Distance: ', self.metric_score['Perfect Match Distance'])
        return 