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

class ErmMatch(BaseAlgo):
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

                wasserstein_loss=torch.tensor(0.0).to(self.cuda)
                erm_loss= torch.tensor(0.0).to(self.cuda) 
                if epoch > self.args.penalty_s:
                    # To cover the varying size of the last batch for data_match_tensor_split, label_match_tensor_split
                    total_batch_size= len(data_match_tensor_split)
                    if batch_idx >= total_batch_size:
                        break
                    curr_batch_size= data_match_tensor_split[batch_idx].shape[0]

        #             data_match= data_match_tensor[idx].to(self.cuda)
                    data_match= data_match_tensor_split[batch_idx].to(self.cuda)
                    data_match= data_match.view( data_match.shape[0]*data_match.shape[1], data_match.shape[2], data_match.shape[3], data_match.shape[4] )            
                    feat_match= self.phi( data_match )
            
        #             label_match= label_match_tensor[idx].to(cuda)           
                    label_match= label_match_tensor_split[batch_idx].to(self.cuda)
                    label_match= label_match.view( label_match.shape[0]*label_match.shape[1] )
                
                    erm_loss+= F.cross_entropy(feat_match, label_match.long()).to(self.cuda)
                    penalty_erm+= float(erm_loss)           
                    
                    train_acc+= torch.sum(torch.argmax(feat_match, dim=1) == label_match ).item()
                    train_size+= label_match.shape[0]
                        
                    # Creating tensor of shape ( domain size, total domains, feat size )
                    if len(feat_match.shape) == 4:
                        feat_match= feat_match.view( curr_batch_size, len(self.train_domains), feat_match.shape[1]*feat_match.shape[2]*feat_match.shape[3] )
                    else:
                         feat_match= feat_match.view( curr_batch_size, len(self.train_domains), feat_match.shape[1] )

                    label_match= label_match.view( curr_batch_size, len(self.train_domains) )

            #             print(feat_match.shape)
                    data_match= data_match.view( curr_batch_size, len(self.train_domains), data_match.shape[1], data_match.shape[2], data_match.shape[3] )    

                    #Positive Match Loss
                    pos_match_counter=0
                    for d_i in range(feat_match.shape[1]):
        #                 if d_i != base_domain_idx:
        #                     continue
                        for d_j in range(feat_match.shape[1]):
                            if d_j > d_i:                        
                                if self.args.pos_metric == 'l2':
                                    wasserstein_loss+= torch.sum( torch.sum( (feat_match[:, d_i, :] - feat_match[:, d_j, :])**2, dim=1 ) ) 
                                elif self.args.pos_metric == 'l1':
                                    wasserstein_loss+= torch.sum( torch.sum( torch.abs(feat_match[:, d_i, :] - feat_match[:, d_j, :]), dim=1 ) )        
                                elif self.args.pos_metric == 'cos':
                                    wasserstein_loss+= torch.sum( cosine_similarity( feat_match[:, d_i, :], feat_match[:, d_j, :] ) )

                                pos_match_counter += feat_match.shape[0]

                    wasserstein_loss = wasserstein_loss / pos_match_counter
                    penalty_ws+= float(wasserstein_loss)                            
                
                    if epoch >= self.args.match_interrupt and self.args.match_flag==1:
                        loss_e += ( self.args.penalty_ws*( epoch - self.args.penalty_s - self.args.match_interrupt )/(self.args.epochs - self.args.penalty_s - self.args.match_interrupt) )*wasserstein_loss
                    else:
                        loss_e += ( self.args.penalty_ws*( epoch- self.args.penalty_s )/(self.args.epochs - self.args.penalty_s) )*wasserstein_loss

                    loss_e += erm_loss
                        
                loss_e.backward(retain_graph=False)
                self.opt.step()
                
                del erm_loss
                del wasserstein_loss 
                del loss_e
                torch.cuda.empty_cache()
                        
   
            print('Train Loss Basic : ',  penalty_erm, penalty_ws )
            print('Train Acc Env : ', 100*train_acc/train_size )
            print('Done Training for epoch: ', epoch)
            
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
            
            if epoch > 0 and epoch % 5==0 and self.args.model_name == 'domain_bed_mnist':
                lr=self.args.lr/(2**(int(epoch/5)))
                print('Learning Rate Scheduling; New LR: ', lr)                
                self.opt= optim.SGD([
                         {'params': filter(lambda p: p.requires_grad, self.phi.parameters()) }, 
                ], lr= lr, weight_decay= self.args.weight_decay, momentum= 0.9,  nesterov=True )     
                