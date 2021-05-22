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

class PACSAugEval(BaseDataLoader):
    def __init__(self, args, list_domains, root, transform=None, data_case='train', match_func=False):
        
        super().__init__(args, list_domains, root, transform, data_case, match_func) 
        self.data, self.labels, self.domains, self.indices, self.objects = self._get_data()

    def _get_data(self):
        
        data_dir= self.root
        to_pil = transforms.ToPILImage()
        to_tensor = transforms.ToTensor()
        
        # Choose subsets that should be included into the training
        list_img = {'aug':[], 'org':[] }
        list_labels = {'aug':[], 'org':[] }
        list_idx= {'aug':[], 'org':[] }
        list_size= {'aug':0, 'org':0 }
        list_classes={'aug':[], 'org':[] }
        
       
        to_tensor=  transforms.Compose([
            transforms.Resize((self.args.img_w, self.args.img_h)),
            transforms.RandomResizedCrop(self.args.img_w, scale=(0.7,1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),                
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
            
        to_tensor_org=  transforms.Compose([
                        transforms.Resize((self.args.img_w, self.args.img_h)),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                      ])
        
        image_counter=0
        for domain in self.list_domains:
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

            pacs_img_trans_org= torch.zeros((pacs_imgs.shape[0], self.args.img_c, self.args.img_w, self.args.img_h))

            for i in range(len(pacs_imgs)):
                curr_img= Image.fromarray( pacs_imgs[i,:,:,:].astype('uint8'), 'RGB' )                
                pacs_img_trans[i,:,:,:]= to_tensor(curr_img)                    
                pacs_img_trans_org[i,:,:,:]= to_tensor_org(curr_img)                  
                
                pacs_idx.append(image_counter)
                image_counter+=1

            print('Source Domain ', domain)
            list_img['aug'].append(pacs_img_trans)
            list_img['org'].append(pacs_img_trans_org)
            
            list_labels['aug'].append(torch.tensor(pacs_labels).long())
            list_labels['org'].append(torch.tensor(pacs_labels).long())
            
            list_idx['aug'].append( pacs_idx )            
            list_idx['org'].append( pacs_idx )            
            
            list_size['aug']+= len(pacs_imgs)
            list_size['org']+= len(pacs_imgs)            
            
            list_classes['aug'].append( len(np.unique(pacs_labels)) )
            list_classes['org'].append( len(np.unique(pacs_labels)) )
        
        if self.match_func:
            print('Match Function Updates')
            num_classes= 7
            for y_c in range(num_classes):
                for key in ['aug', 'org']:
                    base_class_size=0
                    base_class_idx=-1
                    
                    curr_class_size=0
                    for d_idx, domain in enumerate( self.list_domains ):
                        class_idx= list_labels[key][d_idx] == y_c
                        curr_class_size+= list_labels[key][d_idx][class_idx].shape[0]
                    
                    if base_class_size < curr_class_size:
                        base_class_size= curr_class_size
                        if key == 'aug':
                            base_class_idx= 0
                        else:
                            base_class_idx= 1                            
                    
                self.base_domain_size += base_class_size
                print('Max Class Size: ', base_class_size, ' Base Domain Idx: ', base_class_idx, ' Class Label: ', y_c )        
               
        # Stack
        data_imgs = torch.cat(list_img['aug'] + list_img['org'] )
        data_labels = torch.cat(list_labels['aug'] + list_labels['org'] )
        data_indices = np.array(list_idx['aug']+list_idx['org']) 
        data_indices= np.hstack(data_indices)
        list_classes= list_classes['aug'] + list_classes['org']
        self.training_list_size = [ list_size['aug'], list_size['org'] ]           

        #No ground truth objects in PACS, for reference we set them same as data indices
        data_objects= copy.deepcopy(data_indices)        
        
        # Create domain labels
        data_domains = torch.zeros(data_labels.size())
        domain_start=0
        for idx in range(len(self.training_list_size)):
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

        # Convert to onehot
        out_classes= list_classes[0]
        y = torch.eye(out_classes)
        data_labels = y[data_labels]

        # Convert to onehot
        d = torch.eye(len(self.training_list_size))
        data_domains = d[data_domains]
        
        # If shape (B,H,W) change it to (B,C,H,W) with C=1
        if len(data_imgs.shape)==3:
            data_imgs= data_imgs.unsqueeze(1)
            
        print('Shape: Data ', data_imgs.shape, ' Labels ', data_labels.shape, ' Domains ', data_domains.shape, ' Indices ', data_indices.shape, ' Objects ', data_objects.shape)
        return data_imgs, data_labels, data_domains, data_indices, data_objects
