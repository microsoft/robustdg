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
    def __init__(self, args, train_dataset, test_dataset, train_domains, total_domains, domain_size, training_list_size, base_res_dir, run, cuda):
        self.args= args
        self.train_dataset= train_dataset
        self.test_dataset= test_dataset
        self.train_domains= train_domains
        self.total_domains= total_domains
        self.domain_size= domain_size 
        self.training_list_size= training_list_size
        self.base_res_dir= base_res_dir
        self.run= run
        self.cuda= cuda
        
        self.post_string= str(self.args.penalty_ws) + '_' + str(self.args.penalty_diff_ctr) + '_' + str(self.args.match_case) + '_' + str(self.args.match_interrupt) + '_' + str(self.args.match_flag) + '_' + str(self.run) + '_' + self.args.pos_metric + '_' + self.args.model_name
        
        self.phi= self.get_model()
        self.opt= self.get_opt()
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt, step_size=25)    
        
        self.final_acc=[]
        self.val_acc=[]
                
    
    def get_model(self):
        
        if self.args.model_name == 'lenet':
            from models.lenet import LeNet5
            phi= LeNet5()
        if self.args.model_name == 'alexnet':
            from models.alexnet import alexnet
            phi= alexnet(self.args.out_classes, self.args.pre_trained, self.args.method_name)
        if 'resnet' in self.args.model_name:
            from models.resnet import get_resnet
            phi= get_resnet(self.args.model_name, self.args.out_classes, self.args.method_name, 
                            self.args.img_c, self.args.pre_trained)
            
        print('Model Architecture: ', self.args.model_name)
        phi=phi.to(self.cuda)
        return phi
    
    def save_model(self):
        # Store the weights of the model
        torch.save(self.phi.state_dict(), self.base_res_dir + '/Model_' + self.post_string + '.pth')
    
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
            else:
                temp_1, temp_2, indices_matched, perfect_match_rank= get_matched_pairs( self.args, self.cuda, self.train_dataset, self.domain_size, self.total_domains, self.training_list_size, self.phi, self.args.match_case, inferred_match )                
        else:
            inferred_match=0
            data_match_tensor, label_match_tensor, indices_matched, perfect_match_rank= get_matched_pairs( self.args, self.cuda, self.train_dataset, self.domain_size, self.total_domains, self.training_list_size, self.phi, self.args.match_case, inferred_match )
        
        return data_match_tensor, label_match_tensor

    def get_test_accuracy(self):
        
        #Test Env Code
        test_acc= 0.0
        test_size=0

        for batch_idx, (x_e, y_e ,d_e, idx_e) in enumerate(self.test_dataset):
            with torch.no_grad():
                x_e= x_e.to(self.cuda)
                y_e= torch.argmax(y_e, dim=1).to(self.cuda)
                d_e = torch.argmax(d_e, dim=1).numpy()       

                #Forward Pass
                out= self.phi(x_e)                
                
                test_acc+= torch.sum( torch.argmax(out, dim=1) == y_e ).item()
                test_size+= y_e.shape[0]

        print(' Accuracy: ', 100*test_acc/test_size )         
        
        return 100*test_acc/test_size