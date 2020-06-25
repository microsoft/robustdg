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

class ErmMatch(BaseAlgo):
    def __init__(self, train_dataset, data_match_tensor, label_match_tensor, phi, opt, opt_ws, scheduler, epoch, base_domain_idx, bool_erm, bool_ws, bool_ctr):
        
        super().__init__() 
              
    def train(self):
        for epoch in range(epochs):    
            
            penalty_erm=0
            penalty_ws=0
            train_acc= 0.0
            train_size=0
    
            perm = torch.randperm(data_match_tensor.size(0))            
            data_match_tensor_split= torch.split(data_match_tensor[perm], args.batch_size, dim=0)
            label_match_tensor_split= torch.split(label_match_tensor[perm], args.batch_size, dim=0)
            print('Split Matched Data: ', len(data_match_tensor_split), data_match_tensor_split[0].shape, len(label_match_tensor_split))
    
            #Batch iteration over single epoch
            for batch_idx, (x_e, y_e ,d_e, idx_e) in enumerate(train_dataset):
        #         print('Batch Idx: ', batch_idx)

                opt.zero_grad()

                x_e= x_e.to(cuda)
                y_e= torch.argmax(y_e, dim=1).to(cuda)
                d_e= torch.argmax(d_e, dim=1).numpy()

                wasserstein_loss=torch.tensor(0.0).to(cuda)
                erm_loss= torch.tensor(0.0).to(cuda) 
                if epoch > anneal_iter:
                    # To cover the varying size of the last batch for data_match_tensor_split, label_match_tensor_split
                    total_batch_size= len(data_match_tensor_split)
                    if batch_idx >= total_batch_size:
                        break
                    curr_batch_size= data_match_tensor_split[batch_idx].shape[0]

        #             data_match= data_match_tensor[idx].to(cuda)
                    data_match= data_match_tensor_split[batch_idx].to(cuda)
                    data_match= data_match.view( data_match.shape[0]*data_match.shape[1], data_match.shape[2], data_match.shape[3], data_match.shape[4] )            
                    feat_match= phi( data_match )
            
        #             label_match= label_match_tensor[idx].to(cuda)           
                    label_match= label_match_tensor_split[batch_idx].to(cuda)
                    label_match= label_match.view( label_match.shape[0]*label_match.shape[1] )
        
                    erm_loss_2+= erm_loss(feat_match, label_match)
                    penalty_erm_2+= float(erm_loss_2)                
            
                    if args.method_name=="rep_match":
                        temp_out= phi.predict_conv_net( data_match )
                        temp_out= temp_out.view(-1, temp_out.shape[1]*temp_out.shape[2]*temp_out.shape[3])
                        feat_match= phi.predict_fc_net(temp_out)
                        del temp_out
            
                    # Creating tensor of shape ( domain size, total domains, feat size )
                    if len(feat_match.shape) == 4:
                        feat_match= feat_match.view( curr_batch_size, len(train_domains), feat_match.shape[1]*feat_match.shape[2]*feat_match.shape[3] )
                    else:
                         feat_match= feat_match.view( curr_batch_size, len(train_domains), feat_match.shape[1] )

                    label_match= label_match.view( curr_batch_size, len(train_domains) )

            #             print(feat_match.shape)
                    data_match= data_match.view( curr_batch_size, len(train_domains), data_match.shape[1], data_match.shape[2], data_match.shape[3] )    

                    #Positive Match Loss
                    pos_match_counter=0
                    for d_i in range(feat_match.shape[1]):
        #                 if d_i != base_domain_idx:
        #                     continue
                        for d_j in range(feat_match.shape[1]):
                            if d_j > d_i:                        

                                if args.erm_phase:
                                    wasserstein_loss+= torch.sum( torch.sum( (feat_match[:, d_i, :] - feat_match[:, d_j, :])**2, dim=1 ) ) 
                                else:
                                    if args.pos_metric == 'l2':
                                        wasserstein_loss+= torch.sum( torch.sum( (feat_match[:, d_i, :] - feat_match[:, d_j, :])**2, dim=1 ) ) 
                                    elif args.pos_metric == 'l1':
                                        wasserstein_loss+= torch.sum( torch.sum( torch.abs(feat_match[:, d_i, :] - feat_match[:, d_j, :]), dim=1 ) )        
                                    elif args.pos_metric == 'cos':
                                        wasserstein_loss+= torch.sum( cosine_similarity( feat_match[:, d_i, :], feat_match[:, d_j, :] ) )

                                pos_match_counter += feat_match.shape[0]

                    wasserstein_loss = wasserstein_loss / pos_match_counter
                    penalty_ws+= float(wasserstein_loss)                            
                
                    if epoch >= args.match_interrupt and args.match_flag==1:
                        loss_e += ( args.penalty_ws*( epoch - anneal_iter - args.match_interrupt )/(args.epochs - anneal_iter - args.match_interrupt) )*wasserstein_loss
                    else:
                        loss_e += ( args.penalty_ws*( epoch-anneal_iter )/(args.epochs -anneal_iter) )*wasserstein_loss

                    loss_e += args.penalty_erm*erm_loss
                        
                loss_e.backward(retain_graph=False)
                opt.step()
                
                del erm_loss
                del wasserstein_loss 
                del loss_e
                torch.cuda.empty_cache()
        
                train_acc+= torch.sum(torch.argmax(out, dim=1) == y_e ).item()
                train_size+= y_e.shape[0]
                
   
            print('Train Loss Basic : ',  penalty_erm, penalty_ws )
            print('Train Acc Env : ', 100*train_acc/train_size )
            print('Done Training for epoch: ', epoch)
