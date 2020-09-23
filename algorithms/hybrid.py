import sys
import numpy as np
import argparse
import copy
import random
import json
import os

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
from utils.match_function import get_matched_pairs

class Hybrid(BaseAlgo):
    def __init__(self, args, train_dataset, val_dataset, test_dataset, base_res_dir, post_string, cuda):
        
        super().__init__(args, train_dataset, val_dataset, test_dataset, base_res_dir, post_string, cuda) 
        
        self.ctr_save_post_string= str(self.args.match_case) + '_' + str(self.args.match_interrupt) + '_' + str(self.args.match_flag) + '_' + str(self.run) + '_' + self.args.model_name
        self.ctr_load_post_string= str(self.args.ctr_match_case) + '_' + str(self.args.ctr_match_interrupt) + '_' + str(self.args.ctr_match_flag) + '_' + str(self.run) + '_' + self.args.ctr_model_name
                    
    def save_model_erm_phase(self, run):
        
        if not os.path.exists(self.base_res_dir + '/' + self.ctr_load_post_string):
            os.makedirs(self.base_res_dir + '/' + self.ctr_load_post_string)         
                
        # Store the weights of the model
        torch.save(self.phi.state_dict(), self.base_res_dir + '/' + self.ctr_load_post_string + '/Model_' + self.post_string + '_' + str(run) + '.pth')
    
    def init_erm_phase(self):
            
            if self.args.ctr_model_name == 'lenet':
                from models.lenet import LeNet5
                ctr_phi= LeNet5().to(self.cuda)
            if self.args.ctr_model_name == 'alexnet':
                from models.alexnet import alexnet
                ctr_phi= alexnet(self.args.out_classes, self.args.pre_trained, 'matchdg_ctr').to(self.cuda)
            if self.args.ctr_model_name == 'resnet18':
                from models.resnet import get_resnet
                fc_layer=0                
                ctr_phi= get_resnet('resnet18', self.args.out_classes, fc_layer, self.args.img_c, self.args.pre_trained).to(self.cuda)

            # Load MatchDG CTR phase model from the saved weights
            if self.args.os_env:
                base_res_dir=os.getenv('PT_DATA_DIR') + '/' + self.args.dataset_name + '/' + 'matchdg_ctr' + '/' + self.args.ctr_match_layer + '/' + 'train_' + str(self.args.train_domains)             
            else:
                base_res_dir="results/" + self.args.dataset_name + '/' + 'matchdg_ctr' + '/' + self.args.ctr_match_layer + '/' + 'train_' + str(self.args.train_domains)             
            save_path= base_res_dir + '/Model_' + self.ctr_load_post_string + '.pth'
            ctr_phi.load_state_dict( torch.load(save_path) )
            ctr_phi.eval()

            #Inferred Match Case
            if self.args.match_case == -1:
                inferred_match=1
                data_match_tensor, label_match_tensor, indices_matched, perfect_match_rank= get_matched_pairs( self.args, self.cuda, self.train_dataset, self.domain_size, self.total_domains, self.training_list_size, ctr_phi, self.args.match_case, inferred_match )
            # x% percentage match initial strategy
            else:
                inferred_match=0
                data_match_tensor, label_match_tensor, indices_matched, perfect_match_rank= get_matched_pairs( self.args, self.cuda, self.train_dataset, self.domain_size, self.total_domains, self.training_list_size, ctr_phi, self.args.match_case, inferred_match )
                
            return data_match_tensor, label_match_tensor
            
            
    def train(self):
        
        for run_erm in range(self.args.n_runs_matchdg_erm):            
            for epoch in range(self.args.epochs):    
                
                if epoch ==0:
                    data_match_tensor, label_match_tensor= self.init_erm_phase()            
                elif epoch % self.args.match_interrupt == 0 and self.args.match_flag:
                    data_match_tensor, label_match_tensor= self.get_match_function(epoch)

                penalty_erm=0
                penalty_erm_extra=0
                penalty_ws=0
                penalty_aug=0
                train_acc= 0.0
                train_size=0

                perm = torch.randperm(data_match_tensor.size(0))            
                data_match_tensor_split= torch.split(data_match_tensor[perm], self.args.batch_size, dim=0)
                label_match_tensor_split= torch.split(label_match_tensor[perm], self.args.batch_size, dim=0)
                print('Split Matched Data: ', len(data_match_tensor_split), data_match_tensor_split[0].shape, len(label_match_tensor_split))

                #Batch iteration over single epoch
                for batch_idx, (x_e, x_org_e, y_e ,d_e, idx_e) in enumerate(self.train_dataset):
            #         print('Batch Idx: ', batch_idx)

                    self.opt.zero_grad()
                    loss_e= torch.tensor(0.0).to(self.cuda)

                    x_e= x_e.to(self.cuda)
                    x_org_e= x_org_e.to(self.cuda)
                    y_e= torch.argmax(y_e, dim=1).to(self.cuda)
                    d_e= torch.argmax(d_e, dim=1).numpy()

                    #Forward Pass
                    out= self.phi(x_e)
                    erm_loss_extra= F.cross_entropy(out, y_e.long()).to(self.cuda)
                    penalty_erm_extra += float(erm_loss_extra)
                    
                    #Perfect Match on Augmentations
                    out_org= self.phi(x_org_e)
#                     diff_indices= out != out_org
#                     out= out[diff_indices]
#                     out_org= out_org[diff_indices]
                    augmentation_loss=torch.tensor(0.0).to(self.cuda)
                    if self.args.pos_metric == 'l2':
                        augmentation_loss+= torch.sum( torch.sum( (out -out_org)**2, dim=1 ) ) 
                    elif self.args.pos_metric == 'l1':
                        augmentation_loss+= torch.sum( torch.sum( torch.abs(out -out_org), dim=1 ) )        
                    elif self.args.pos_metric == 'cos':
                        augmentation_loss+= torch.sum( cosine_similarity( out, out_org ) )

                    augmentation_loss = augmentation_loss / out.shape[0]
#                     print('Augmented Images Fraction: ', out.shape, self.args.batch_size, augmentation_loss)
                    penalty_aug+= float(augmentation_loss)                            

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

            #             label_match= label_match_tensor[idx].to(self.cuda)           
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


                        loss_e += ( self.args.penalty_ws*( epoch- self.args.penalty_s )/(self.args.epochs - self.args.penalty_s) )*wasserstein_loss
                        loss_e += self.args.penalty_aug*augmentation_loss
                        loss_e += erm_loss
                        loss_e += erm_loss_extra
                        

                    loss_e.backward(retain_graph=False)
                    self.opt.step()

                    del erm_loss_extra
                    del erm_loss
                    del wasserstein_loss 
                    del loss_e
                    torch.cuda.empty_cache()

                print('Train Loss Basic : ', penalty_erm_extra, penalty_aug, penalty_erm, penalty_ws )
                print('Train Acc Env : ', 100*train_acc/train_size )
                print('Done Training for epoch: ', epoch)    
                
                #Val Dataset Accuracy
                self.val_acc.append( self.get_test_accuracy('val') )

                #Test Dataset Accuracy
                self.final_acc.append( self.get_test_accuracy('test') )                    
            # Save the model's weights post training
            self.save_model_erm_phase(run_erm)
