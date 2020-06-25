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


class BaseAlgo():
    def __init__(self, args, dataset_name, list_train_domains, root, transform=None, data_case='train'):
        self.args= args        
        self.rep_dim= args.rep_dim
        self.num_classes= args.out_classes
        self.pre_trained= args.pre_trained
        self.epochs=args.epochs
        self.batch_size=args.batch_size
        self.learning_rate= args.lr
        self.lmd=args.penalty_w
        self.anneal_iter= args.penalty_s
        self.match_flag=args.match_flag
        self.match_interrupt=args.match_interrupt
        self.base_domain_idx= args.base_domain_idx
        
        self.phi= self.get_model()
        self.opt= self.get_opt()
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt, step_size=25)    
        
        self.final_acc=[]
        self.val_acc=[]
                
    
    def get_model(self):
        
        if self.args.dataset == 'rot_mnist' or self.args.dataset == 'fashion_mnist':
            if self.args.model_name == 'lenet':
                from models.LeNet import *
            else:
                from models.ResNet import *                
                num_ch=1
                pre_trained=0
                
        elif self.args.dataset == 'pacs':
            if self.args.model_name == 'alexnet':
                from models.AlexNet import *
                num_ch=3
                pre_trained=1                
            elif self.args.model_name == 'resnet18':
                from models.ResNet import *
                num_ch=3
                pre_trained=1
        
        if self.args.model_name == 'lenet':
            phi= LeNet5().to(cuda)
        elif self.args.model_name == 'alexnet':
            phi= alexnet(self.num_classes, pre_trained, self.args.erm_base ).to(cuda)            
        elif self.args.model_name == 'resnet18':
            phi= get_resnet('resnet18', self.num_classes, self.args.erm_base, num_ch, pre_trained).to(cuda)
                                        
#         else:
#             rep_dim=512
#             phi= get_resnet('resnet18', rep_dim, self.args.erm_base, num_ch, pre_trained).to(cuda)
                    
        print('Model Archtecture: ', args.model_name)
        return phi
    
    
    def get_opt(self):
        if self.args.opt == 'sgd':
            opt= optim.SGD([
                         {'params': filter(lambda p: p.requires_grad, phi.parameters()) }, 
                ], lr= learning_rate, weight_decay= 5e-4, momentum= 0.9,  nesterov=True )        
        elif self.args.opt == 'adam':
            opt= optim.Adam([
                        {'params': filter(lambda p: p.requires_grad, phi.parameters())},
                ], lr= learning_rate)
        
        return opt

    
    def get_match_function(self, epoch):
        #Start initially with randomly defined batch; else find the local approximate batch
        if epoch % match_interrupt == 0:
            if epoch > 0:                    
                inferred_match=1
                if args.match_flag:
                    data_match_tensor, label_match_tensor, indices_matched, perfect_match_rank= get_matched_pairs( args, train_dataset, domain_size, total_domains, training_list_size, phi, args.match_case, inferred_match )
                    match_counter+=1
                else:
                    temp_1, temp_2, indices_matched, perfect_match_rank= get_matched_pairs( args, train_dataset, domain_size, total_domains, training_list_size, phi, args.match_case, inferred_match )                
            else:
                inferred_match=0
                data_match_tensor, label_match_tensor, indices_matched, perfect_match_rank= get_matched_pairs( args, train_dataset, domain_size, total_domains, training_list_size, phi, args.match_case, inferred_match )
        
        return data_match_tensor, label_match_tensor
