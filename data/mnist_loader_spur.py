#Common imports
import os
import random
import copy
import numpy as np

#Pytorch
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms

#Base Class
from .data_loader import BaseDataLoader

class MnistRotated(BaseDataLoader):
    def __init__(self, args, list_domains, mnist_subset, root, transform=None, data_case='train', match_func=False, download=True):
        
        super().__init__(args, list_domains, root, transform, data_case, match_func) 
        self.mnist_subset = mnist_subset
        self.download = download
        
        self.data, self.labels, self.domains, self.indices, self.objects, self.spur = self._get_data()

    def _get_data(self):
                
        # Choose subsets that should be included into the training
        list_img = []
        list_labels = []
        list_idx= []
        list_size= []
        list_spur= []
        data_dir= self.root + self.args.dataset_name + '_' + self.args.mnist_case + '/'           

        for domain in self.list_domains:
            load_dir= data_dir + self.data_case + '/' + 'seed_' + str(self.mnist_subset) + '_domain_' + str(domain)

            #Augmentation
            if self.data_case =='train' and self.args.mnist_aug:                
                mnist_imgs= torch.load( load_dir +  '_data.pt')
            else:
                mnist_imgs= torch.load( load_dir +  '_org_data.pt')
            mnist_labels= torch.load( load_dir +  '_label.pt')
            mnist_idx= list(range(len(mnist_imgs)))
            
            mnist_spur= np.load(load_dir +  '_spur.npy')

            print('Source Domain ', domain)
            list_img.append(mnist_imgs)
            list_labels.append(mnist_labels)
            list_idx.append(mnist_idx)
            list_size.append(mnist_imgs.shape[0])
            list_spur.append(mnist_spur)
             
        if self.match_func:
            print('Match Function Updates')
            num_classes= 10
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
                   
        # Stack
        data_imgs = torch.cat(list_img)
        data_labels = torch.cat(list_labels)
        data_indices = np.array(list_idx)
        data_indices= np.hstack(data_indices)
        self.training_list_size= list_size
        
        # Spurious Color Labels
        data_spur = np.array(list_spur)
        data_spur= np.hstack(data_spur)                
        
        #Rotated MNIST the objects are same the data indices
        data_objects= copy.deepcopy(data_indices)
        
        # Create domain labels
        data_domains = torch.zeros(data_labels.size())
        domain_start=0
        for idx in range(len(self.list_domains)):
            curr_domain_size= self.training_list_size[idx]
            data_domains[ domain_start: domain_start+ curr_domain_size ] += idx
            domain_start+= curr_domain_size        

        # Shuffle everything one more time
        inds = np.arange(data_labels.size()[0])
        np.random.shuffle(inds)
        data_imgs = data_imgs[inds]
        data_labels = data_labels[inds]
        data_domains = data_domains[inds].long()
        data_indices = data_indices[inds]
        data_objects = data_objects[inds]
        data_spur = data_spur[inds]

        # Convert to onehot
        y = torch.eye(10)
        data_labels = y[data_labels]

        # Convert to onehot
        d = torch.eye(len(self.list_domains))
        data_domains = d[data_domains]
        
        # If shape (B,H,W) change it to (B,C,H,W) with C=1
        if len(data_imgs.shape)==3:
            data_imgs= data_imgs.unsqueeze(1)        
        
        print('Shape: Data ', data_imgs.shape, ' Labels ', data_labels.shape, ' Domains ', data_domains.shape, ' Indices ', data_indices.shape, ' Objects ', data_objects.shape, ' Spur Corr ', data_spur.shape)
        return data_imgs, data_labels, data_domains, data_indices, data_objects, data_spur
