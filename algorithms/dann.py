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
import torch.autograd as autograd

from .algo import BaseAlgo
from utils.helper import l1_dist, l2_dist, embedding_dist, cosine_similarity

class DANN(BaseAlgo):
    def __init__(self, args, train_dataset, val_dataset, test_dataset, base_res_dir, post_string, cuda):
        
        super().__init__(args, train_dataset, val_dataset, test_dataset, base_res_dir, post_string, cuda) 
        
        self.conditional = bool(self.args.conditional)
        self.class_balance = False        
        
        self.featurizer = self.phi.feat_net
        self.classifier = self.phi.fc
        self.discriminator = self.phi.disc
        self.class_embeddings = self.phi.embedding
        
        self.grad_penalty= self.args.grad_penalty
        self.lambda_= self.args.penalty_ws
        self.d_steps_per_g_step= self.args.d_steps_per_g_step
        self.initial_lr= 0.01
        
        # Optimizers
        self.disc_opt = torch.optim.SGD(
            (list(self.discriminator.parameters()) + 
                list(self.class_embeddings.parameters())),
            lr=self.initial_lr,
            weight_decay=5e-4)

        self.gen_opt = torch.optim.SGD(
            (list(self.featurizer.parameters()) + 
                list(self.classifier.parameters())),
            lr=self.initial_lr,
            weight_decay=5e-4)     
        
    def train(self):
        
        self.max_epoch=-1
        self.max_val_acc=0.0
        for epoch in range(self.args.epochs):   
                    
            penalty_erm=0
            penalty_dann=0
            train_acc= 0.0
            train_size=0
                    
            #Batch iteration over single epoch
            for batch_idx, (x_e, y_e ,d_e, idx_e, obj_e) in enumerate(self.train_dataset):
        #         print('Batch Idx: ', batch_idx)

                x_e= x_e.to(self.cuda)
                y_e= torch.argmax(y_e, dim=1).to(self.cuda)
                d_e= torch.argmax(d_e, dim=1).to(self.cuda)
        
                all_x = x_e
                all_y = y_e
                all_z = self.featurizer(all_x)
                if self.conditional:
                    disc_input = all_z + self.class_embeddings(all_y)
                else:
                    disc_input = all_z
                disc_out = self.discriminator(disc_input)
                disc_labels = d_e        
            
                if self.class_balance:
                    y_counts = F.one_hot(all_y).sum(dim=0)
                    weights = 1. / (y_counts[all_y] * y_counts.shape[0]).float()
                    disc_loss = F.cross_entropy(disc_out, disc_labels, reduction='none')
                    disc_loss = (weights * disc_loss).sum()
                else:
                    disc_loss = F.cross_entropy(disc_out, disc_labels)

                disc_softmax = F.softmax(disc_out, dim=1)
                input_grad = autograd.grad(disc_softmax[:, disc_labels].sum(),
                    [disc_input], create_graph=True)[0]
                grad_penalty = (input_grad**2).sum(dim=1).mean(dim=0)
                                
                #Disc Loss
                disc_loss += self.grad_penalty * grad_penalty

                #Gen Loss
                all_preds = self.classifier(all_z)
                classifier_loss = F.cross_entropy(all_preds, all_y)
                gen_loss = (classifier_loss +
                            (self.lambda_ * -disc_loss))

                penalty_erm += float(classifier_loss)
                penalty_dann += float(disc_loss)
                
                d_steps_per_g = self.d_steps_per_g_step
                if (epoch % (1+d_steps_per_g) < d_steps_per_g):
                    self.disc_opt.zero_grad()
                    disc_loss.backward()
                    self.disc_opt.step()
                else:
                    self.disc_opt.zero_grad()
                    self.gen_opt.zero_grad()
                    gen_loss.backward()
                    self.gen_opt.step()
                
                del classifier_loss
                del gen_loss 
                del disc_loss
                torch.cuda.empty_cache()
                
                #Forward Pass
                features = self.featurizer(x_e)
                out = self.classifier(features)                
                train_acc+= torch.sum(torch.argmax(out, dim=1) == y_e ).item()
                train_size+= y_e.shape[0]                
                        
   
            print('Train Loss Basic : ',  penalty_erm, penalty_dann )
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