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

class Erm(BaseAlgo):
    def __init__(self, args, train_dataset, val_dataset, test_dataset, base_res_dir, post_string, cuda):
        
        super().__init__(args, train_dataset, val_dataset, test_dataset, base_res_dir, post_string, cuda) 
              
    def train(self):
        
        self.max_epoch=-1
        self.max_val_acc=0.0        
        for epoch in range(self.args.epochs):   
            
            if epoch ==0 or (epoch % self.args.match_interrupt == 0 and self.args.match_flag):
                data_match_tensor, label_match_tensor= self.get_match_function(epoch)
            
            penalty_erm=0
            penalty_ws=0
            train_acc= 0.0
            train_size=0
    
            perm = torch.randperm(data_match_tensor.size(0))            
            data_match_tensor_split= torch.split(data_match_tensor[perm], self.args.batch_size, dim=0)
            label_match_tensor_split= torch.split(label_match_tensor[perm], self.args.batch_size, dim=0)
            print('Split Matched Data: ', len(data_match_tensor_split), data_match_tensor_split[0].shape, len(label_match_tensor_split))
    
            #Batch iteration over single epoch
            for batch_idx, (x_e, y_e ,d_e, idx_e) in enumerate(self.train_dataset):
        #         print('Batch Idx: ', batch_idx)

                self.opt.zero_grad()
                loss_e= torch.tensor(0.0).to(self.cuda)
                
                x_e= x_e.to(self.cuda)
                y_e= torch.argmax(y_e, dim=1).to(self.cuda)
                d_e= torch.argmax(d_e, dim=1).numpy()
                
                #Forward Pass
                out= self.phi(x_e)
                erm_loss= F.cross_entropy(out, y_e.long()).to(self.cuda)
                loss_e+= erm_loss
                penalty_erm += float(loss_e)

                #Backprorp
                loss_e.backward(retain_graph=False)
                self.opt.step()
                
                del erm_loss
                del loss_e
                torch.cuda.empty_cache()
        
                train_acc+= torch.sum(torch.argmax(out, dim=1) == y_e ).item()
                train_size+= y_e.shape[0]
                
   
            print('Train Loss Basic : ',  penalty_erm )
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
            
        # Save the model's weights post training
        self.save_model()