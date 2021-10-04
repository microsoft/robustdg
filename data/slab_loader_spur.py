#Common imports
import os
import random
import copy
import numpy as np
import h5py
from PIL import Image

#Sklearn
from scipy.stats import bernoulli

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
    def __init__(self, args, list_train_domains, root, transform=None, data_case='train', match_func=False, base_size=10000, freq_ratio=50, data_dim=2, total_slabs=5, seed=0, mask_linear=0):
        
        super().__init__(args, list_train_domains, root, transform, data_case, match_func) 
        
        self.base_size = base_size
        self.freq_ratio= freq_ratio
        self.data_dim= data_dim
        self.total_slabs= total_slabs
        self.slab_noise= args.slab_noise
        self.seed= seed
        self.mask_linear= mask_linear
        
        print(list_train_domains)
        if self.data_case == 'train':        
            self.domain_size = len(list_train_domains)*[self.base_size]
            # Default Train Domains: [0.0, 0.10]            
            self.spur_probs= [ float(domain) for domain in list_train_domains ]
            
        elif self.data_case == 'val':
            self.domain_size = len(list_train_domains)*[int(self.base_size/4)]
            self.spur_probs= [ float(domain) for domain in list_train_domains ]
            
        elif self.data_case == 'test':        
            self.domain_size = len(list_train_domains)*[self.base_size]
            self.spur_probs= [ float(domain) for domain in list_train_domains ]
        
        print('\n')
        print('Data Case: ', self.data_case)
        
        self.base_domain_size= self.base_size
        self.train_data, self.train_labels, self.train_domain, self.train_indices, self.train_spur = self._get_data(self.domain_size, self.data_dim, self.total_slabs, self.spur_probs, self.slab_noise, self.data_case, self.seed, mask_linear)        

    def _get_data(self, domain_size, data_dim, total_slabs, spur_probs, slab_noise, data_case, seed, mask_linear):

        list_data = []
        list_labels = []
        list_objs= []
        list_spur= []
        total_domains= len(domain_size)

        for idx in range(total_domains):

            num_samples= domain_size[idx]        
            spur_prob= spur_probs[idx]

            _, data, labels, match_obj= get_data(num_samples, spur_prob, slab_noise, total_slabs, data_case, seed, mask_linear)
            print('Source Domain: ', idx, ' Size: ', data.shape, labels.shape, match_obj.shape)
            
            if self.data_case == 'test':
                rand_var= bernoulli.rvs(0.0, size=labels.shape[0])
            else:
                rand_var= bernoulli.rvs(0.0, size=labels.shape[0])
                
            spur_data=np.ones((labels.shape[0], 1))
            spur_indices=[]
            
            #Linear feature as the spurious feature
            for l_idx in range(labels.shape[0]):
#                 if data[l_idx, 0] > 0:
#                     spur_indices.append(1)
#                 else:
#                     spur_indices.append(0)                    
            
                if (data[l_idx, 0] > 0 and data[l_idx, 0] < 0.5) or (data[l_idx, 0] < -0.5 and data[l_idx, 0] > -1.0) :
                    spur_indices.append(1)
                else:
                    spur_indices.append(0)                    
            
# TODO: Additional spurious feature             
#             for l_idx in range(labels.shape[0]):
#                 label= labels[l_idx]
#                 spur_indices.append(label)
#                 if label:
#                     spur_data[l_idx]= np.random.uniform(0.1, 3.0, 1)[0]
#                 else:
#                     spur_data[l_idx]= np.random.uniform(-3.0, -0.1, 1)[0]                            
                
#                 #Adding noise to the spurious feature - label relationship
#                 if rand_var[l_idx]:
#                     spur_data[l_idx]= -1*spur_data[l_idx]
#                     spur_indices[l_idx]= -1*spur_indices[l_idx]
                
#             data= np.concatenate((data, spur_data), axis=1)
            
            list_data.append(torch.tensor(data))
            list_labels.append(torch.tensor(labels))
            list_objs.append(match_obj)
            list_spur.append(spur_indices)
        
        # Stack data from the different domains
        data_feat = torch.cat(list_data)
        data_labels = torch.cat(list_labels)
        data_objs= np.hstack(list_objs)
        data_spur= np.hstack(list_spur)

        # Create domain labels
        data_domains = torch.zeros(data_labels.size())
        domain_start=0
        for idx in range(total_domains):
            curr_domain_size= domain_size[idx]
            data_domains[ domain_start: domain_start+ curr_domain_size ] += idx
            domain_start+= curr_domain_size

        # Shuffle everything one more time
    #     inds = np.arange(data_labels.size()[0])
    #     np.random.shuffle(inds)
    #     data_feat = data_feat[inds]
    #     data_labels = data_labels[inds].long()
    #     data_domains = data_domains[inds].long()

        # Convert to onehot
        y = torch.eye(2)
        data_labels = y[data_labels]
#         # Convert to onehot
#         d = torch.eye(len(self.list_train_domains))
#         data_domains = d[data_domains]
    
        #Type Casting
        data_feat= data_feat.type(torch.FloatTensor)
        data_labels = data_labels.long()
        data_domains = data_domains.long()

        print('Final Dataset: ', data_feat.shape, data_labels.shape, data_domains.shape, data_objs.shape, data_spur.shape)
        return data_feat, data_labels, data_domains, data_objs, data_spur