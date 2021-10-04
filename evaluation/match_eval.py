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

from .base_eval import BaseEval
from utils.match_function import get_matched_pairs

class MatchEval(BaseEval):
    
    def __init__(self, args, train_dataset, val_dataset, test_dataset, base_res_dir, run, cuda):
        super().__init__(args, train_dataset, val_dataset, test_dataset, base_res_dir, run, cuda)
                
    def get_metric_eval(self):
        
        if self.args.match_func_data_case=='train':
            dataset= self.train_dataset['data_loader']
            total_domains= self.train_dataset['total_domains']
            domain_list= self.train_dataset['domain_list']
            base_domain_size= self.train_dataset['base_domain_size']
            domain_size_list= self.train_dataset['domain_size_list']
            
        elif self.args.match_func_data_case== 'val':
            dataset= self.val_dataset['data_loader']
            total_domains= self.val_dataset['total_domains']
            domain_list= self.val_dataset['domain_list']
            base_domain_size= self.val_dataset['base_domain_size']
            domain_size_list= self.val_dataset['domain_size_list']
        
        elif self.args.match_func_data_case== 'test':
            dataset= self.test_dataset['data_loader']
            total_domains= self.test_dataset['total_domains']
            domain_list= self.test_dataset['domain_list']
            base_domain_size= self.test_dataset['base_domain_size']
            domain_size_list= self.test_dataset['domain_size_list']
    
        inferred_match=1        
        
        # Self Augmentation Match Function evaluation will always follow perfect matches
        if self.args.match_func_aug_case:
            perfect_match= 1
        else:
            perfect_match= self.args.perfect_match
            
        data_matched, domain_data, perfect_match_rank= get_matched_pairs( self.args, self.cuda, dataset, base_domain_size, total_domains, domain_size_list, self.phi, self.args.match_case, perfect_match, inferred_match )
        
        perfect_match_rank= np.array(perfect_match_rank)            

        self.metric_score['Perfect Match Score']= 100*np.sum( perfect_match_rank < 1 )/perfect_match_rank.shape[0]
        self.metric_score['TopK Perfect Match Score']= 100*np.sum( perfect_match_rank < self.args.top_k )/perfect_match_rank.shape[0]
        self.metric_score['Perfect Match Rank']= np.mean(perfect_match_rank)
        
        print('Perfect Match Score: ', self.metric_score['Perfect Match Score']   )                    
        print('TopK Perfect Match Score: ', self.metric_score['TopK Perfect Match Score'] )          
        print('Perfect Match Rank: ', self.metric_score['Perfect Match Rank'] )            
#         print('Perfect Match Distance: ', self.metric_score['Perfect Match Distance'])
        return 



        # Perfect Match Prediction Discrepancy
#         perm = torch.randperm(data_match_tensor.size(0))
#         data_match_tensor_split= torch.split(data_match_tensor[perm], self.args.batch_size, dim=0)
#         label_match_tensor_split= torch.split(label_match_tensor[perm], self.args.batch_size, dim=0)
        
#         total_batches= len(data_match_tensor_split)
#         penalty_ws= 0.0
#         for batch_idx in range(total_batches):
            
#             curr_batch_size= data_match_tensor_split[batch_idx].shape[0]
            
#             data_match= data_match_tensor_split[batch_idx].to(self.cuda)
#             data_match= data_match.view( data_match.shape[0]*data_match.shape[1], data_match.shape[2], data_match.shape[3], data_match.shape[4] )            
#             feat_match= self.phi( data_match )

#             label_match= label_match_tensor_split[batch_idx].to(self.cuda)
#             label_match= label_match.view( label_match.shape[0]*label_match.shape[1] )

#             # Creating tensor of shape ( domain size, total domains, feat size )
#             if len(feat_match.shape) == 4:
#                 feat_match= feat_match.view( curr_batch_size, len(total_domains), feat_match.shape[1]*feat_match.shape[2]*feat_match.shape[3] )
#             else:
#                  feat_match= feat_match.view( curr_batch_size, len(total_domains), feat_match.shape[1] )

#             label_match= label_match.view( curr_batch_size, len(total_domains) )

#     #             print(feat_match.shape)
#             data_match= data_match.view( curr_batch_size, len(total_domains), data_match.shape[1], data_match.shape[2], data_match.shape[3] )    

#             #Positive Match Loss
#             pos_match_counter=0
#             for d_i in range(feat_match.shape[1]):
# #                 if d_i != base_domain_idx:
# #                     continue
#                 for d_j in range(feat_match.shape[1]):
#                     if d_j > d_i:                        
#                         if self.args.pos_metric == 'l2':
#                             wasserstein_loss+= torch.sum( torch.sum( (feat_match[:, d_i, :] - feat_match[:, d_j, :])**2, dim=1 ) ) 
#                         elif self.args.pos_metric == 'l1':
#                             wasserstein_loss+= torch.sum( torch.sum( torch.abs(feat_match[:, d_i, :] - feat_match[:, d_j, :]), dim=1 ) )        
#                         elif self.args.pos_metric == 'cos':
#                             wasserstein_loss+= torch.sum( cosine_similarity( feat_match[:, d_i, :], feat_match[:, d_j, :] ) )

#                         pos_match_counter += feat_match.shape[0]

#             wasserstein_loss = wasserstein_loss / pos_match_counter
#             penalty_ws+= float(wasserstein_loss)                            
        
#         self.metric_score['Perfect Match Distance']= penalty_ws