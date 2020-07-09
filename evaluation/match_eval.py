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
from utils.match_function import get_matched_pairs, perfect_match_score

class MatchEval(BaseEval):
    
    def __init__(self, args, train_dataset, test_dataset, train_domains, total_domains, domain_size, training_list_size, cuda):
        super().__init__(args, train_dataset, test_dataset, train_domains, total_domains, domain_size, training_list_size, cuda)
        
        self.match_top_1=0.0
        self.match_top_k=0.0
        self.match_rank=0.0
        
    def get_match_score(self, top_k):
        
        inferred_match=1
        data_match_tensor, label_match_tensor, indices_matched, perfect_match_rank= get_matched_pairs( self.args, self.cuda, self.train_dataset, self.domain_size, self.total_domains, self.training_list_size, self.phi, self.args.match_case, inferred_match )
        
        score= perfect_match_score(indices_matched)
        perfect_match_rank= np.array(perfect_match_rank)            

        self.match_top_1= score
        self.match_top_k= 100*np.sum( perfect_match_rank <top_k )/perfect_match_rank.shape[0]
        self.match_rank= np.mean(perfect_match_rank)

        print('Perfect Match Score: ', self.match_top_1   )                    
        print('TopK Perfect Match Score: ',  self.match_top_k )          
        print('Perfect Match Rank: ',  self.match_rank )            
        return 