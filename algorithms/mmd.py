import sys
import numpy as np
import argparse
import copy
import random
import json
import time

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

class MMD(BaseAlgo):
    def __init__(self, args, train_dataset, val_dataset, test_dataset, base_res_dir, post_string, cuda):
        
        super().__init__(args, train_dataset, val_dataset, test_dataset, base_res_dir, post_string, cuda) 

        self.mmd_gamma= self.args.penalty_ws
        self.gaussian= bool(self.args.gaussian)
        self.conditional= bool(self.args.conditional)
        if self.gaussian: 
            self.kernel_type = "gaussian"
        else:
            self.kernel_type = "mean_cov"
        
        self.featurizer = self.phi.feat_net
        self.classifier = self.phi.fc
        
        print('Initial Params: ', )
        
    def my_cdist(self, x1, x2):
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.addmm(x2_norm.transpose(-2, -1),
                          x1,
                          x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
        return res.clamp_min_(1e-30)
    
    def gaussian_kernel(self, x, y, gamma=[0.001, 0.01, 0.1, 1, 10, 100,
                                           1000]):
        D = self.my_cdist(x, y)
        K = torch.zeros_like(D)

        for g in gamma:
            K.add_(torch.exp(D.mul(-g)))

        return K

    def mmd(self, x, y):
        if self.kernel_type == "gaussian":
            Kxx = self.gaussian_kernel(x, x).mean()
            Kyy = self.gaussian_kernel(y, y).mean()
            Kxy = self.gaussian_kernel(x, y).mean()
            return Kxx + Kyy - 2 * Kxy
        else:
            mean_x = x.mean(0, keepdim=True)
            mean_y = y.mean(0, keepdim=True)
            cent_x = x - mean_x
            cent_y = y - mean_y
            cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
            cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

            mean_diff = (mean_x - mean_y).pow(2).mean()
            cova_diff = (cova_x - cova_y).pow(2).mean()

            return mean_diff + cova_diff
    
    def mmd_regularization(self, features, d, nmb):
        penalty= torch.tensor(0.0).to(self.cuda)
        for d_i in range(nmb):
            for d_j in range(d_i + 1, nmb):
                f_i= features[ d == d_i ]
                f_j= features[ d == d_j ]
                penalty += self.mmd(f_i, f_j)
        return penalty
        
    def train(self):
        
        self.max_epoch=-1
        self.max_val_acc=0.0
        for epoch in range(self.args.epochs):   
                    
            penalty_erm=0
            penalty_mmd=0
            train_acc= 0.0
            train_size=0
                    
            #Batch iteration over single epoch
            for batch_idx, (x_e, y_e ,d_e, idx_e, obj_e) in enumerate(self.train_dataset):
        #         print('Batch Idx: ', batch_idx)

                self.opt.zero_grad()
                loss_e= torch.tensor(0.0).to(self.cuda)
                
                x_e= x_e.to(self.cuda)
                y_e= torch.argmax(y_e, dim=1).to(self.cuda)
                d_e= torch.argmax(d_e, dim=1)
                
                #Forward Pass
                features = self.featurizer(x_e)
                out = self.classifier(features)                
                
                #ERM
                erm_loss= F.cross_entropy(out, y_e.long()).to(self.cuda)
                loss_e+= erm_loss
                penalty_erm += float(loss_e)
                
                #MMD
                mmd_loss=torch.tensor(0.0).to(self.cuda)
                match_domains= torch.unique(d_e)
                class_labels= torch.unique(y_e)
                nmb = len(match_domains)

                if self.conditional:
                    for y_c in range(len(class_labels)):                    
                        features_c= features[ y_e == y_c ]
                        d_c= d_e[ y_e == y_c ]
                        if len(torch.unique(d_c)) != nmb:
                            print('*********************************')
                            print('Error: Some classes not distributed across all the domains; issues for class conditional methods')
                            continue
                        mmd_loss+= self.mmd_regularization(features_c, d_c, nmb)
                else:
                    mmd_loss+= self.mmd_regularization(features, d_e, nmb)            

                if nmb > 1:
                    mmd_loss /= (nmb * (nmb - 1) / 2)
                                
                penalty_mmd+= float(mmd_loss)
                
                #Backward Pass
                loss_e+= self.mmd_gamma*mmd_loss                
                loss_e.backward(retain_graph=False)
                self.opt.step()
                
                del erm_loss
                del mmd_loss 
                del loss_e
                torch.cuda.empty_cache()
                
                train_acc+= torch.sum(torch.argmax(out, dim=1) == y_e ).item()
                train_size+= y_e.shape[0]                
                        
   
            print('Train Loss Basic : ',  penalty_erm, penalty_mmd )
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