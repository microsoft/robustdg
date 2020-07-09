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

from utils.match_function import get_matched_pairs

class BaseAlgo():
    def __init__(self, args, train_dataset, train_domains, total_domains, domain_size, training_list_size, cuda):
        self.args= args
        self.train_dataset= train_dataset
        self.train_domains= train_domains
        self.total_domains= total_domains
        self.domain_size= domain_size 
        self.training_list_size= training_list_size
        self.cuda= cuda
        self.phi= self.get_model()
        self.opt= self.get_opt()
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt, step_size=25)    
        
        self.final_acc=[]
        self.val_acc=[]
                
    
    def get_model(self):
        
        if self.args.model_name == 'lenet':
            from models.LeNet import LeNet5
            phi= LeNet5().to(self.cuda)
        if self.args.model_name == 'alexnet':
            from models.AlexNet import alexnet
            phi= alexnet(self.args.out_classes, self.args.pre_trained, self.args.method_name).to(self.cuda)
        if self.args.model_name == 'resnet18':
            from models.ResNet import get_resnet
            phi= get_resnet('resnet18', self.args.out_classes, self.args.method_name, self.args.img_c, self.args.pre_trained).to(self.cuda)
        
#         else:
#             rep_dim=512
#             phi= get_resnet('resnet18', self.args.rep_dim, self.args.erm_base, num_ch, pre_trained).to(cuda)
                    
        print('Model Architecture: ', self.args.model_name)
        return phi
    
    
    def get_opt(self):
        if self.args.opt == 'sgd':
            opt= optim.SGD([
                         {'params': filter(lambda p: p.requires_grad, self.phi.parameters()) }, 
                ], lr= self.args.lr, weight_decay= 5e-4, momentum= 0.9,  nesterov=True )        
        elif self.args.opt == 'adam':
            opt= optim.Adam([
                        {'params': filter(lambda p: p.requires_grad, self.phi.parameters())},
                ], lr= self.args.lr)
        
        return opt

    
    def get_match_function(self, epoch):
        #Start initially with randomly defined batch; else find the local approximate batch
        if epoch > 0:                    
            inferred_match=1
            if self.args.match_flag:
                data_match_tensor, label_match_tensor, indices_matched, perfect_match_rank= get_matched_pairs( self.args, self.cuda, self.train_dataset, self.domain_size, self.total_domains, self.training_list_size, self.phi, self.args.match_case, inferred_match )
                match_counter+=1
            else:
                temp_1, temp_2, indices_matched, perfect_match_rank= get_matched_pairs( self.args, self.cuda, self.train_dataset, self.domain_size, self.total_domains, self.training_list_size, self.phi, self.args.match_case, inferred_match )                
        else:
            inferred_match=0
            data_match_tensor, label_match_tensor, indices_matched, perfect_match_rank= get_matched_pairs( self.args, self.cuda, self.train_dataset, self.domain_size, self.total_domains, self.training_list_size, self.phi, self.args.match_case, inferred_match )
        
        return data_match_tensor, label_match_tensor
