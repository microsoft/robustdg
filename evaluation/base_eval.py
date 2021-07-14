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
    def __init__(self, args, train_dataset, val_dataset, test_dataset, base_res_dir, run, cuda):
        
        self.args= args
        self.train_dataset= train_dataset
        self.val_dataset= val_dataset
        self.test_dataset= test_dataset
        
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
        self.metric_score={}                
    
    def get_model(self, run_matchdg_erm=0):
        
        if self.args.model_name == 'lenet':
            from models.lenet import LeNet5
            phi= LeNet5()
            
        if self.args.model_name == 'slab':
            from models.slab import SlabClf
            if self.args.method_name in ['csd', 'matchdg_ctr']:
                fc_layer=0
            else:
                fc_layer= self.args.fc_layer
            phi= SlabClf(self.args.slab_data_dim, self.args.out_classes, fc_layer)
                        
        if self.args.model_name == 'fc':
            from models.fc import FC
            if self.args.method_name in ['csd', 'matchdg_ctr']:
                fc_layer=0
            else:
                fc_layer= self.args.fc_layer
            phi= FC(self.args.out_classes, fc_layer)            
            
        if self.args.model_name == 'domain_bed_mnist':
            from models.domain_bed_mnist import DomainBed
            if self.args.method_name in ['csd', 'matchdg_ctr']:
                fc_layer=0
            else:
                fc_layer= self.args.fc_layer                        
            phi= DomainBed(self.args.img_c, fc_layer)
            
        if self.args.model_name == 'alexnet':
            from models.alexnet import alexnet
            if self.args.method_name in ['csd', 'matchdg_ctr']:
                fc_layer=0
            else:
                fc_layer= self.args.fc_layer            
            phi= alexnet(self.args.model_name, self.args.out_classes, fc_layer, 
                            self.args.img_c, self.args.pre_trained, self.args.os_env)
            
        if 'resnet' in self.args.model_name:
            from models.resnet import get_resnet
            if self.args.method_name in ['csd', 'matchdg_ctr']:
                fc_layer=0
            else:
                fc_layer= self.args.fc_layer
            phi= get_resnet(self.args.model_name, self.args.out_classes, fc_layer, 
                            self.args.img_c, self.args.pre_trained, self.args.dp_noise, self.args.os_env)
            
        if 'densenet' in self.args.model_name:
            from models.densenet import get_densenet
            if self.args.method_name in ['csd', 'matchdg_ctr']:
                fc_layer=0
            else:
                fc_layer= self.args.fc_layer
            phi= get_densenet(self.args.model_name, self.args.out_classes, fc_layer, 
                            self.args.img_c, self.args.pre_trained, self.args.os_env)
                    
        print('Model Architecture: ', self.args.model_name)
        
        self.phi= phi.to(self.cuda)        
        self.load_model(run_matchdg_erm)

        return
    
    def load_model(self, run_matchdg_erm):
        
        if self.args.method_name in ['erm_match', 'csd', 'irm', 'perf_match', 'rand_match', 'mask_linear', 'mmd', 'dann']:
            self.save_path= self.base_res_dir + '/Model_' + self.post_string
                
        elif self.args.method_name == 'matchdg_ctr':
            self.save_path= self.base_res_dir + '/Model_' + self.ctr_save_post_string 
            
        elif self.args.method_name in ['matchdg_erm', 'hybrid']:
            self.save_path=  (
                                self.base_res_dir + '/' + 
                                self.ctr_load_post_string + '/Model_' + 
                                self.post_string + '_' + str(run_matchdg_erm)
                            )
            
        print(self.save_path)        
        self.phi.load_state_dict( torch.load(self.save_path + '.pth') )
        self.phi.eval()      
        
        if self.args.method_name in ['csd', 'csd_slab']:
            self.save_path= self.base_res_dir + '/Sms_' + self.post_string 
            self.sms= torch.load(self.save_path + '.pt')
            
            self.save_path= self.base_res_dir + '/SmBiases_' + self.post_string 
            self.sm_biases= torch.load(self.save_path + '.pt')
        
        return
    
    def forward(self, x_e):
        
        if self.args.method_name in ['csd', 'csd_slab']:
            x_e = self.phi(x_e)        
            w_c, b_c = self.sms[0, :, :], self.sm_biases[0, :]
            logits= torch.matmul(x_e, w_c) + b_c            
        else:
            logits= self.phi(x_e)
        
        return logits
    
    def get_logits(self):

        #Train Environment Logits
        final_out=[]
        for batch_idx, (x_e, y_e ,d_e, idx_e, obj_e) in enumerate(self.train_dataset['data_loader']):
            #Random Shuffling along the batch axis
            x_e= x_e[ torch.randperm(x_e.size()[0]) ]

            with torch.no_grad():
                x_e= x_e.to(self.cuda)
                if self.args.mia_logit:
                    out= self.forward(x_e)
                else:
                    out= F.softmax(self.forward(x_e), dim=1)
                final_out.append(out)            
            
        final_out= torch.cat(final_out)
        print('Train Logits: ', final_out.shape, self.save_path)
        pickle.dump([final_out], open( self.save_path + "_train.pkl", 'wb'))

        #Test Environment Logits
        final_out=[]
        for batch_idx, (x_e, y_e ,d_e, idx_e, obj_e) in enumerate(self.test_dataset['data_loader']):
            #Random Shuffling along the batch axis
            x_e= x_e[ torch.randperm(x_e.size()[0]) ]

            with torch.no_grad():
                x_e= x_e.to(self.cuda)
                if self.args.mia_logit:
                    out= self.forward(x_e)
                else:
                    out= F.softmax(self.forward(x_e), dim=1)
                final_out.append(out)
            
        final_out= torch.cat(final_out)
        print('Test Logits: ', final_out.shape, self.save_path)
        pickle.dump([final_out], open( self.save_path + "_test.pkl", 'wb'))
    
        return
    
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
        
        test_acc= 0.0
        test_size=0
        for batch_idx, (x_e, y_e ,d_e, idx_e, obj_e) in enumerate(dataset):
            with torch.no_grad():
                x_e= x_e.to(self.cuda)
                y_e= torch.argmax(y_e, dim=1).to(self.cuda)

                #Forward Pass
                out= self.forward(x_e)                
                loss_e= torch.mean(F.cross_entropy(out, y_e.long()).to(self.cuda))

                test_acc+= torch.sum( torch.argmax(out, dim=1) == y_e ).item()
                test_size+= y_e.shape[0]

        print(' Accuracy: ', 100*test_acc/test_size ) 
        self.metric_score[case +' accuracy']= 100*test_acc/test_size  

        return 

    