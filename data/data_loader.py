import os
import copy
import numpy as np
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms

class BaseDataLoader(data_utils.Dataset):
    def __init__(self, args, list_domains, root, transform=None, data_case='train', match_func=False):
        self.args= args
        self.list_domains = list_domains
        if self.args.os_env:
            self.root = os.getenv('PT_DATA_DIR') + root
        else:
            self.root = 'data/datasets' + root
        self.transform = transform
        self.data_case = data_case
        self.match_func= match_func
        
        self.base_domain_size= 0
        self.list_size=[]
        self.data= [] 
        self.labels= [] 
        self.domains= [] 
        self.indices= [] 
        self.objects= []

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]
        d = self.domains[index]
        idx = self.indices[index]
        objs= self.objects[index]
            
        if self.transform is not None:
            x = self.transform(x)
        return x, y, d, idx, objs

    def get_size(self):
        return self.labels.shape[0]
    
    def get_item_spur(self, index):
        x = self.data[index]
        y = self.labels[index]
        d = self.domains[index]
        idx = self.indices[index]
        objs= self.objects[index]
        spur = self.spur[index]
            
        if self.transform is not None:
            x = self.transform(x)
        return x, y, d, idx, objs, spur        
