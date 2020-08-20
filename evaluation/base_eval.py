import sys
import numpy as np
import pandas as pd
import argparse
import copy
import random
import json
import pickle

import torch
from torch.autograd import grad
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import torch.utils.data as data_utils

class BaseEval():
    def __init__(self, args, train_dataset, test_dataset, train_domains, 
                 total_domains, domain_size, training_list_size, base_res_dir, run, cuda):
        
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
        
        # Model save paths depending on the method: ERM_Match, MatchDG_CTR, MatchDG_ERM
        self.post_string=   (
                            str(self.args.penalty_ws) + '_' + 
                            str(self.args.penalty_diff_ctr) + '_' + 
                            str(self.args.match_case) + '_' + 
                            str(self.args.match_interrupt) + '_' + 
                            str(self.args.match_flag) + '_' + 
                            str(self.run) + '_' + 
                            self.args.pos_metric + '_' + 
                            self.args.model_name
                            )
        
        self.ctr_save_post_string=  (
                                    str(self.args.match_case) + '_' + 
                                    str(self.args.match_interrupt) + '_' + 
                                    str(self.args.match_flag) + '_' + 
                                    str(self.run) + '_' + 
                                    self.args.model_name
                                    )
        
        self.ctr_load_post_string=  (
                                    str(self.args.ctr_match_case) + '_' + 
                                    str(self.args.ctr_match_interrupt) + '_' + 
                                    str(self.args.ctr_match_flag) + '_' + 
                                    str(self.run) + '_' + 
                                    self.args.ctr_model_name
                                    )
        
        if self.args.method_name in ['erm_match', 'erm', 'irm', 'dro']:
            self.save_path= self.base_res_dir + '/Model_' + self.post_string
                
        elif self.args.method_name == 'matchdg_ctr':
            self.save_path= self.base_res_dir + '/Model_' + self.ctr_save_post_string 
            
        elif self.args.method_name == 'matchdg_erm':
            self.save_path=  (
                                self.base_res_dir + '/' + 
                                self.ctr_load_post_string + '/Model_' + 
                                self.post_string + '_' + str(run)
                            )
                
        self.phi= self.get_model()        
        self.load_model()
        self.metric_score={}                
    
    def get_model(self):
        
        if self.args.model_name == 'lenet':
            from models.lenet import LeNet5
            phi= LeNet5()
        if self.args.model_name == 'alexnet':
            from models.alexnet import alexnet
            phi= alexnet(self.args.out_classes, self.args.pre_trained, self.args.method_name)
        if self.args.model_name == 'domain_bed_mnist':
            from models.domain_bed_mnist import DomainBed
            phi= DomainBed( self.args.img_c )
        if 'resnet' in self.args.model_name:
            from models.resnet import get_resnet
            phi= get_resnet(self.args.model_name, self.args.out_classes, self.args.method_name, 
                            self.args.img_c, self.args.pre_trained)
        
        print('Model Architecture: ', self.args.model_name)
        phi= phi.to(self.cuda)
        return phi
    
    def load_model(self):
                
        self.phi.load_state_dict( torch.load(self.save_path + '.pth') )
        self.phi.eval()        
    
    def get_logits(self):

        #Train Environment Logits
        final_out=[]
        for batch_idx, (x_e, y_e ,d_e, idx_e) in enumerate(self.train_dataset):
            #Random Shuffling along the batch axis
            x_e= x_e[ torch.randperm(x_e.size()[0]) ]

            with torch.no_grad():
                x_e= x_e.to(self.cuda)
                if self.args.mia_logit:
                    out= self.phi(x_e)
                else:
                    out= F.softmax(self.phi(x_e), dim=1)
                final_out.append(out)            
            
        final_out= torch.cat(final_out)
        print('Train Logits: ', final_out.shape, self.save_path)
        pickle.dump([final_out], open( self.save_path + "_train.pkl", 'wb'))

        #Test Environment Logits
        final_out=[]
        for batch_idx, (x_e, y_e ,d_e, idx_e) in enumerate(self.test_dataset):
            #Random Shuffling along the batch axis
            x_e= x_e[ torch.randperm(x_e.size()[0]) ]

            with torch.no_grad():
                x_e= x_e.to(self.cuda)
                if self.args.mia_logit:
                    out= self.phi(x_e)
                else:
                    out= F.softmax(self.phi(x_e), dim=1)
                final_out.append(out)
            
        final_out= torch.cat(final_out)
        print('Test Logits: ', final_out.shape, self.save_path)
        pickle.dump([final_out], open( self.save_path + "_test.pkl", 'wb'))
    
        return
    
    def get_metric_eval(self):
        
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
                loss_e= torch.mean(F.cross_entropy(out, y_e.long()).to(self.cuda))
                
                test_acc+= torch.sum( torch.argmax(out, dim=1) == y_e ).item()
                test_size+= y_e.shape[0]

        print(' Accuracy: ', 100*test_acc/test_size ) 
        self.metric_score['Test Accuracy']= 100*test_acc/test_size  
        return 

    