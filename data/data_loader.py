import os
import random
import copy
import numpy as np
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms

class BaseDataLoader(data_utils.Dataset):
    def __init__(self, args, list_train_domains, root, transform=None, data_case='train', match_func=False):
        self.args= args
        self.list_train_domains = list_train_domains
        if self.args.os_env:
            self.root = os.getenv('PT_DATA_DIR') + root
        else:
            self.root = 'data/datasets' + root
        self.transform = transform
        self.data_case = data_case
        self.match_func= match_func
        
        self.base_domain_size= 0
        self.training_list_size=[]
        self.train_data= [] 
        self.train_labels= [] 
        self.train_domain= [] 
        self.train_indices= [] 

    def __len__(self):
        return self.train_labels.shape[0]

    def __getitem__(self, index):
        x = self.train_data[index]
        y = self.train_labels[index]
        d = self.train_domain[index]
        idx = self.train_indices[index]
            
        if self.transform is not None:
            x = self.transform(x)
        return x, y, d, idx

    def get_size(self):
        return self.train_labels.shape[0]
    
    def get_item_spur(self, index):
        x = self.train_data[index]
        y = self.train_labels[index]
        d = self.train_domain[index]
        idx = self.train_indices[index]
        spur = self.train_spur[index]
            
        if self.transform is not None:
            x = self.transform(x)
        return x, y, d, idx, spur
