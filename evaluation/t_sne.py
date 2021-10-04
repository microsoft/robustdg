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
from utils.helper import t_sne_plot

class TSNE(BaseEval):
    
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
        
        t_sne_label={}
        for y_c in range(self.args.out_classes):
            t_sne_label[y_c]=[]

        t_sne_domain={}
        for domain in range(total_domains):
            t_sne_domain[domain]=[]

        feat_all=[]
        label_all=[]
        domain_all=[]


        for batch_idx, (x_e, y_e, d_e, idx_e, obj_e) in enumerate(dataset):
            x_e= x_e.to(self.cuda)
            y_e= torch.argmax(y_e, dim=1)
            d_e= torch.argmax(d_e, dim=1)

            with torch.no_grad():
                feat_all.append( self.phi(x_e).cpu() )
                label_all.append( y_e )
                domain_all.append( d_e )

        feat_all= torch.cat(feat_all)
        label_all= torch.cat(label_all).numpy()
        domain_all= torch.cat(domain_all).numpy()

        #t-SNE plots     
        t_sne_out= t_sne_plot( feat_all ).tolist() 
#             if args.rep_dim > 2:
#                 t_sne_out= t_sne_plot( feat_all ).tolist() 
#             elif args.rep_dim ==2:
#                 t_sne_out = feat_all.detach().numpy().tolist() 
#             else:
#                 print('Issue: Represenation Dimension cannot be less than 2')

        #print('T-SNE', np.array(t_sne_out).shape, feat_all.shape, label_all.shape, domain_all.shape)

        for idx in range(feat_all.shape[0]):
            key= label_all[idx]
            t_sne_label[key].append( t_sne_out[idx] )

        with open(self.save_path + '_label.json', 'w') as fp:
            json.dump(t_sne_label, fp)                        

        for idx in range(feat_all.shape[0]):
            key= domain_all[idx]
            t_sne_domain[key].append( t_sne_out[idx] )

        with open(self.save_path + '_domain.json', 'w') as fp:
            json.dump(t_sne_domain, fp)                        

        ##Current output of T-SNE is passed as None as we don't need to repot any average metric value
        self.metric_score['T-SNE Label']= {}
        self.metric_score['T-SNE Domain']= {}

        return         