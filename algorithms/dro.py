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

class DRO(BaseAlgo):
    def __init__(self, args, train_dataset, val_dataset, test_dataset, train_domains, total_domains, domain_size, training_list_size, base_res_dir, post_string, cuda):
        
        super().__init__(args, train_dataset, val_dataset, test_dataset, train_domains, total_domains, domain_size, training_list_size, base_res_dir, post_string, cuda) 
              
    def train(self):
        
        for epoch in range(self.args.epochs):   
            
            if epoch ==0 or (epoch % self.args.match_interrupt == 0 and self.args.match_flag):
                data_match_tensor, label_match_tensor= self.get_match_function(epoch)
            
            penalty_erm=0
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

                erm_loss= torch.tensor(0.0).to(self.cuda) 
                if epoch > self.args.penalty_s:
                    # To cover the varying size of the last batch for data_match_tensor_split, label_match_tensor_split
                    total_batch_size= len(data_match_tensor_split)
                    if batch_idx >= total_batch_size:
                        break
                    curr_batch_size= data_match_tensor_split[batch_idx].shape[0]
                    
                    data_match= data_match_tensor_split[batch_idx].to(self.cuda)
                    label_match= label_match_tensor_split[batch_idx].to(self.cuda)
                    
                    for domain_idx in range(data_match.shape[1]):
                        
                        data_idx= data_match[:,domain_idx,:,:,:]            
                        feat_idx= self.phi( data_idx )

                        label_idx= label_match[:, domain_idx]
                        label_idx= label_idx.view(label_idx.shape[0])
                        erm_loss = torch.max(erm_loss, F.cross_entropy(feat_idx, label_idx.long()).to(self.cuda))
                        
                    penalty_erm+= float(erm_loss)                    
                    loss_e += erm_loss
                        
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

        # Save the model's weights post training
        self.save_model()