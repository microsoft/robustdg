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
from utils.helper import l1_dist, l2_dist, embedding_dist, cosine_similarity, slab_batch_process

class PerfMatch(BaseAlgo):
    def __init__(self, args, train_dataset, val_dataset, test_dataset, base_res_dir, post_string, cuda):
        
        super().__init__(args, train_dataset, val_dataset, test_dataset, base_res_dir, post_string, cuda) 
              
    def train(self):
        
        self.max_epoch=-1
        self.max_val_acc=0.0        
        for epoch in range(self.args.epochs):   
            
            penalty_erm=0
            penalty_ws=0
            train_acc= 0.0
            train_size=0
    
            #Batch iteration over single epoch
            for batch_idx, (x_e, y_e ,d_e, idx_e) in enumerate(self.train_dataset):
        #         print('Batch Idx: ', batch_idx)
                
                #Process current batch as per slab dataset
#                 x_e, y_e ,d_e, idx_e= slab_batch_process(x_e, y_e ,d_e, idx_e)            
            
                self.opt.zero_grad()
                loss_e= torch.tensor(0.0).to(self.cuda)
                
                x_e= x_e.to(self.cuda)
                y_e= torch.argmax(y_e, dim=1).to(self.cuda)
                
                #Forward Pass
                out= self.phi(x_e)
                erm_loss= F.cross_entropy(out, y_e.long()).to(self.cuda)
                loss_e+= erm_loss
                penalty_erm += float(loss_e)

                #Perfect Match Penalty
                ws_loss= torch.tensor(0.0).to(self.cuda)
                counter=0
                match_objs= np.unique(idx_e)
                feat= self.phi.feat_net(x_e)
                for obj in match_objs:
                    indices= idx_e == obj
                    feat_obj= feat[indices]
                    d_obj= d_e[indices]

                    match_domains= torch.unique(d_obj)

                    if len(match_domains) != len(torch.unique(d_e)):
        #                 print('Error: Positivty Violation, objects not present in all the domains')
                        continue

                    for d_i in range(len(match_domains)):
                        for d_j in range(len(match_domains)):
                            if d_j <= d_i:
                                continue
                            x1= feat_obj[ d_obj == d_i ]
                            x2= feat_obj[ d_obj == d_j ]

                            #Typecasting
        #                     print(x1.shape, x2.shape)
                            x1= x1.view(x1.shape[0], 1, x1.shape[1])
                            ws_loss= torch.sum( torch.sum( torch.sum( (x1 -x2)**2, dim=2), dim=1 ) )
        #                     ws_loss= torch.sum( torch.sum( torch.sum( torch.abs(x1 -x2), dim=2), dim=1 ) )
                            counter+= x1.shape[0]*x2.shape[0]

                ws_loss= ws_loss/counter
                penalty_ws += float(ws_loss)                
                                
                #Backprop
                loss_e+= self.args.penalty_ws*ws_loss*((epoch+1)/self.args.epochs)
                loss_e.backward(retain_graph=False)
                self.opt.step()
                
                del erm_loss
                del loss_e
                torch.cuda.empty_cache()
        
                train_acc+= torch.sum(torch.argmax(out, dim=1) == y_e ).item()
                train_size+= y_e.shape[0]
                
   
            print('Train Loss Basic : ',  penalty_erm, penalty_ws )
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