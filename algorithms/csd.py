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

from .algo import BaseAlgo
from utils.helper import l1_dist, l2_dist, embedding_dist, cosine_similarity


class CSD(BaseAlgo):
    def __init__(self, args, train_dataset, val_dataset, test_dataset, base_res_dir, post_string, cuda):
        
        super().__init__(args, train_dataset, val_dataset, test_dataset, base_res_dir, post_string, cuda)         
        
        # H_dim as per the feature layer dimension of ResNet-18
        H_dim= self.args.rep_dim
        self.K, m, self.num_classes = 1, H_dim, self.args.out_classes 
        num_domains = self.total_domains

        self.sms = torch.nn.Parameter(torch.normal(0, 1e-1, size=[self.K+1, m, self.num_classes], dtype=torch.float, device='cuda:0'), requires_grad=True)
        self.sm_biases = torch.nn.Parameter(torch.normal(0, 1e-1, size=[self.K+1, self.num_classes], dtype=torch.float, device='cuda:0'), requires_grad=True)
    
        self.embs = torch.nn.Parameter(torch.normal(mean=0., std=1e-1, size=[num_domains, self.K], dtype=torch.float, device='cuda:0'), requires_grad=True)
        self.cs_wt = torch.nn.Parameter(torch.normal(mean=.1, std=1e-4, size=[], dtype=torch.float, device='cuda:0'), requires_grad=True)

        self.opt= optim.SGD([
                         {'params': filter(lambda p: p.requires_grad, self.phi.parameters()) },
                         {'params': self.sms },
                         {'params': self.sm_biases },
                         {'params': self.embs },
                         {'params': self.cs_wt }
                ], lr= self.args.lr, weight_decay= 5e-4, momentum= 0.9,  nesterov=True )          
        
        self.criterion = torch.nn.CrossEntropyLoss()
        
    def forward(self, x, y, di, eval_case=0):
        x = self.phi(x)        
        w_c, b_c = self.sms[0, :, :], self.sm_biases[0, :]
        logits_common = torch.matmul(x, w_c) + b_c
       
        if eval_case:
            return logits_common
 
        domains= di
        c_wts = torch.matmul(domains, self.embs)
    
        # B x K
        batch_size = x.shape[0]
        c_wts = torch.cat((torch.ones((batch_size, 1), dtype=torch.float).to(self.cuda)*self.cs_wt, c_wts), 1)
        c_wts = torch.tanh(c_wts).to(self.cuda)
        w_d, b_d = torch.einsum("bk,krl->brl", c_wts, self.sms), torch.einsum("bk,kl->bl", c_wts, self.sm_biases)
        logits_specialized = torch.einsum("brl,br->bl", w_d, x) + b_d

        specific_loss = self.criterion(logits_specialized, y)
        class_loss = self.criterion(logits_common, y)

        sms = self.sms
        diag_tensor = torch.stack([torch.eye(self.K+1).to(self.cuda) for _ in range(self.num_classes)], dim=0)
        cps = torch.stack([torch.matmul(sms[:, :, _], torch.transpose(sms[:, :, _], 0, 1)) for _ in range(self.num_classes)], dim=0)
        orth_loss = torch.mean((1-diag_tensor)*(cps - diag_tensor)**2)

        loss = class_loss + specific_loss + orth_loss 
        return loss, class_loss, logits_common
    
    def epoch_callback(self, nepoch, final=False):
        if nepoch % 100 == 0:
            print (self.embs, torch.norm(self.sms[0]), torch.norm(self.sms[1]))
                          
    def train(self):
        
        self.max_epoch=-1
        self.max_val_acc=0.0
        for epoch in range(self.args.epochs):   
            
            penalty_erm=0
            penalty_csd=0
            train_acc= 0.0
            train_size=0
    
            #Batch iteration over single epoch
            for batch_idx, (x_e, y_e ,d_e, idx_e, obj_e) in enumerate(self.train_dataset):
        #         print('Batch Idx: ', batch_idx)

                self.opt.zero_grad()
                loss_e= torch.tensor(0.0).to(self.cuda)
                
                x_e= x_e.to(self.cuda)
                y_e= torch.argmax(y_e, dim=1).to(self.cuda)
                
                #Forward Pass
                csd_loss, erm_loss, out= self.forward(x_e, y_e, d_e.to(self.cuda), eval_case=0)
                loss_e+= csd_loss
                penalty_csd += float(loss_e)
                penalty_erm += float(erm_loss)

                #Backprorp
                loss_e.backward(retain_graph=False)
                self.opt.step()
                
                del csd_loss
                del loss_e
                torch.cuda.empty_cache()
        
                train_acc+= torch.sum(torch.argmax(out, dim=1) == y_e ).item()
                train_size+= y_e.shape[0]
                
   
            print('Train Loss Basic : ',  penalty_erm, penalty_csd - penalty_erm )
            print('Train Acc Env : ', 100*train_acc/train_size )
            print('Done Training for epoch: ', epoch)
            
            #Train Dataset Accuracy
            self.train_acc.append( 100*train_acc/train_size )
            
            #Val Dataset Accuracy
            self.val_acc.append( self.get_test_accuracy('val') )
            
            #Test Dataset Accuracy
            self.final_acc.append( self.get_test_accuracy('test') )

            #Save the model if current best epoch as per validation loss
            if self.val_acc[-1] > self.max_val_acc:
                self.max_val_acc=self.val_acc[-1]
                self.max_epoch= epoch
                self.save_model()                
                
            print('Current Best Epoch: ', self.max_epoch, ' with Test Accuracy: ', self.final_acc[self.max_epoch])



    def get_test_accuracy(self, case):
        
        #Test Env Code
        test_acc= 0.0
        test_size=0
        if case == 'val':
            dataset= self.val_dataset
        elif case == 'test':
            dataset= self.test_dataset

        for batch_idx, (x_e, y_e ,d_e, idx_e, obj_e) in enumerate(dataset):
            with torch.no_grad():
                x_e= x_e.to(self.cuda)
                y_e= torch.argmax(y_e, dim=1).to(self.cuda)

                #Forward Pass
                out= self.forward(x_e, y_e, d_e.to(self.cuda), eval_case=1)
                
                test_acc+= torch.sum( torch.argmax(out, dim=1) == y_e ).item()
                test_size+= y_e.shape[0]
                
        print(' Accuracy: ', case,  100*test_acc/test_size )         
        
        return 100*test_acc/test_size
    
    def save_model(self):
        # Store the weights of the model
        torch.save(self.phi.state_dict(), self.base_res_dir + '/Model_' + self.post_string + '.pth')
        # Store the parameters
        torch.save(self.sms, self.base_res_dir + '/Sms_' + self.post_string + ".pt")
        torch.save(self.sm_biases, self.base_res_dir + '/SmBiases_' + self.post_string + ".pt")