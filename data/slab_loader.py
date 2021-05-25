    #Common imports
import os
import random
import copy
import numpy as np
import h5py
from PIL import Image

#Pytorch
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms
from torchvision import datasets, transforms

#Base Class
from .data_loader import BaseDataLoader

#Specific Modules 
from utils.slab_data import *


class SlabData(BaseDataLoader):
    def __init__(self, args, list_domains, root, transform=None, data_case='train', match_func=False, base_size=10000, freq_ratio=50, data_dim=2, total_slabs=5, seed=0, mask_linear=0):
        
        super().__init__(args, list_domains, root, transform, data_case, match_func) 
        
        self.base_size = base_size
        self.freq_ratio= freq_ratio
        self.data_dim= data_dim
        self.total_slabs= total_slabs
        self.slab_noise= args.slab_noise
        self.seed= seed
        self.mask_linear= mask_linear
        
        if args.method_name == 'mask_linear':
            self.mask_linear= 1
        else:
            self.mask_linear= 0
        
        if self.data_case == 'train':        
            self.domain_size = len(self.list_domains)*[self.base_size]
            # Default Train Domains: [0.0, 0.10]            
            self.spur_probs= [ float(domain) for domain in self.list_domains ]
            
        elif self.data_case == 'val':
            self.domain_size = len(self.list_domains)*[int(self.base_size/4)]
            self.spur_probs= [ float(domain) for domain in self.list_domains ]
            
        elif self.data_case == 'test':        
            self.domain_size = len(self.list_domains)*[self.base_size]
            self.spur_probs= [ float(domain) for domain in self.list_domains ]
        
        print('\n')
        print('Data Case: ', self.data_case)
        
        self.data, self.labels, self.domains, self.indices, self.objects = self._get_data(self.domain_size, self.data_dim, self.total_slabs, self.spur_probs, self.slab_noise, self.data_case, self.seed, self.mask_linear)        

    def _get_data(self, domain_size, data_dim, total_slabs, spur_probs, slab_noise, data_case, seed, mask_linear):

        list_data = []
        list_labels = []
        list_idx = []
        list_objs = []
        list_size = []
        total_domains= len(domain_size)

        for idx in range(total_domains):

            num_samples= domain_size[idx]        
            spur_prob= spur_probs[idx]

            _, data, labels, match_obj= get_data(num_samples, spur_prob, slab_noise, total_slabs, data_case, seed, mask_linear)                        
            data_idx= list(range(len(data)))
                
            print('Source Domain: ', idx, ' Size: ', data.shape, labels.shape, match_obj.shape)
            list_data.append(torch.tensor(data))
            list_labels.append(torch.tensor(labels))
            list_idx.append(data_idx)
            list_objs.append(match_obj)
            list_size.append(len(data))

        if self.match_func:
            print('Match Function Updates')
            num_classes= 2
            for y_c in range(num_classes):
                base_class_size=0
                base_class_idx=-1
                for d_idx, domain in enumerate( self.list_domains ):
                    class_idx= list_labels[d_idx] == y_c
                    curr_class_size= list_labels[d_idx][class_idx].shape[0]
                    if base_class_size < curr_class_size:
                        base_class_size= curr_class_size
                        base_class_idx= d_idx
                self.base_domain_size += base_class_size
                print('Max Class Size: ', base_class_size, ' Base Domain Idx: ', base_class_idx, ' Class Label: ', y_c )
                
        # Stack data from the different domains
        data_feat = torch.cat(list_data)
        data_labels = torch.cat(list_labels)
        data_indices = np.array(list_idx)
        data_indices= np.hstack(data_indices)
        self.training_list_size= list_size
        
        data_objects= np.hstack(list_objs)

        # Create domain labels
        data_domains = torch.zeros(data_labels.size())
        domain_start=0
        for idx in range(total_domains):
            curr_domain_size= domain_size[idx]
            data_domains[ domain_start: domain_start+ curr_domain_size ] += idx
            domain_start+= curr_domain_size
            
            
        # Shuffle everything one more time
        inds = np.arange(data_labels.size()[0])
        np.random.shuffle(inds)
        data_feat = data_feat[inds]
        data_labels = data_labels[inds]
        data_domains = data_domains[inds].long()
        data_indices = data_indices[inds]
        data_objects = data_objects[inds]


        # Convert to onehot
        y = torch.eye(2)
        data_labels = y[data_labels]
        
        # Convert to onehot
        d = torch.eye(len(self.list_domains))
        data_domains = d[data_domains]
    
        #Type Casting
        data_feat= data_feat.type(torch.FloatTensor)
        data_labels = data_labels.long()
        
        print('Shape: Data ', data_feat.shape, ' Labels ', data_labels.shape, ' Domains ', data_domains.shape, ' Indices ', data_indices.shape, ' Objects ', data_objects.shape)
        return data_feat, data_labels, data_domains, data_indices, data_objects
