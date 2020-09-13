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

class PACS(BaseDataLoader):
    def __init__(self, args, list_train_domains, root, transform=None, data_case='train', match_func=False):
        
        super().__init__(args, list_train_domains, root, transform, data_case, match_func) 
        self.train_data, self.train_labels, self.train_domain, self.train_indices = self._get_data()

    def _get_data(self):
        
        data_dir= self.root
        to_pil = transforms.ToPILImage()
        to_tensor = transforms.ToTensor()
        
        # Choose subsets that should be included into the training
        training_list_img = []
        training_list_labels = []
        training_list_idx= []
        training_list_size= []
        training_out_classes=[]
       
        if self.data_case == 'train':
            to_tensor=  transforms.Compose([
                transforms.Resize((self.args.img_w, self.args.img_h)),
                transforms.RandomResizedCrop(self.args.img_w, scale=(0.7,1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
                transforms.RandomGrayscale(),                
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
             to_tensor=  transforms.Compose([
                transforms.Resize((self.args.img_w, self.args.img_h)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            

        for domain in self.list_train_domains:
            if self.data_case == 'train':
                domain_data = h5py.File(data_dir + domain + '_' + 'train.hdf5','r')
            elif self.data_case == 'val':
                domain_data = h5py.File(data_dir + domain + '_' + 'val.hdf5','r')
            elif self.data_case == 'test':
                domain_data = h5py.File(data_dir + domain + '_' + 'test.hdf5','r')
            
            pacs_imgs= domain_data.get('images')
            # Convert labels in the range(1,7) into (0,6)
            pacs_labels= np.array(domain_data.get('labels')) -1               
            pacs_idx=[]
            print('Image: ', pacs_imgs.shape, ' Labels: ', pacs_labels.shape, ' Out Classes: ', len(np.unique(pacs_labels)))
            
            pacs_img_trans= torch.zeros((pacs_imgs.shape[0], self.args.img_c, self.args.img_w, self.args.img_h))
            for i in range(len(pacs_imgs)):
                curr_img= Image.fromarray( pacs_imgs[i,:,:,:].astype('uint8'), 'RGB' )
                
                pacs_img_trans[i,:,:,:]= to_tensor(curr_img)                    
                pacs_idx.append(i)

            print('Source Domain ', domain)
            training_list_img.append(pacs_img_trans)
            training_list_labels.append(torch.tensor(pacs_labels).long())
            training_list_idx.append( pacs_idx )
            training_list_size.append(len(pacs_imgs))            
            training_out_classes.append( len(np.unique(pacs_labels)) )
        
        if self.match_func:
            print('Match Function Updates')
            num_classes= 7
            for y_c in range(num_classes):
                base_class_size=0
                base_class_idx=-1
                for d_idx, domain in enumerate( self.list_train_domains ):
                    class_idx= training_list_labels[d_idx] == y_c
                    curr_class_size= training_list_labels[d_idx][class_idx].shape[0]
                    if base_class_size < curr_class_size:
                        base_class_size= curr_class_size
                        base_class_idx= d_idx
                self.base_domain_size += base_class_size
                print('Max Class Size: ', base_class_size, base_class_idx, y_c )
        
                
        # Stack
        train_imgs = torch.cat(training_list_img)
        train_labels = torch.cat(training_list_labels)
        train_indices = np.array(training_list_idx)
        train_indices= np.hstack(train_indices)
        self.training_list_size = training_list_size            
    
        print(train_imgs.shape, train_labels.shape, train_indices.shape)
        print(self.training_list_size)
        
        # Create domain labels
        train_domains = torch.zeros(train_labels.size())
        domain_start=0
        for idx in range(len(self.list_train_domains)):
            curr_domain_size= self.training_list_size[idx]
            train_domains[ domain_start: domain_start+ curr_domain_size ] += idx
            domain_start+= curr_domain_size
           
        # Shuffle everything one more time
        inds = np.arange(train_labels.size()[0])
        np.random.shuffle(inds)
        train_imgs = train_imgs[inds]
        train_labels = train_labels[inds]
        train_domains = train_domains[inds].long()
        train_indices = train_indices[inds]

        # Convert to onehot
        out_classes= training_out_classes[0]
        y = torch.eye(out_classes)
        train_labels = y[train_labels]

        # Convert to onehot
        d = torch.eye(len(self.list_train_domains))
        train_domains = d[train_domains]
        
        print(train_imgs.shape, train_labels.shape, train_domains.shape, train_indices.shape)
        # If shape (B,H,W) change it to (B,C,H,W) with C=1
        if len(train_imgs.shape)==3:
            train_imgs= train_imgs.unsqueeze(1)
        return train_imgs, train_labels, train_domains, train_indices