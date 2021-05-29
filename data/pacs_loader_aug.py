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

class PACSAug(BaseDataLoader):
    def __init__(self, args, list_domains, root, transform=None, data_case='train', match_func=False):
        
        super().__init__(args, list_domains, root, transform, data_case, match_func) 
        self.data, self.data_org, self.labels, self.domains, self.indices, self.objects = self._get_data()

    def __getitem__(self, index):
        x = self.data[index]
        x_org = self.data_org[index]
        y = self.labels[index]
        d = self.domains[index]
        idx = self.indices[index]
        obj = self.objects[index]
            
        if self.transform is not None:
            x = self.transform(x)
        return x, x_org, y, d, idx, obj        
        
    def _get_data(self):
        
        data_dir= self.root
        to_pil = transforms.ToPILImage()
        to_tensor = transforms.ToTensor()
        
        # Choose subsets that should be included into the training
        list_img = []
        list_img_org= []
        list_labels = []
        list_idx= []
        list_size= []
        list_classes=[]
       
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
        
        to_tensor_org=  transforms.Compose([
                transforms.Resize((self.args.img_w, self.args.img_h)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
              ])
  
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
                pacs_idx.append(i)

            print('Source Domain ', domain)
            list_img.append(pacs_img_trans)
            list_img_org.append(pacs_img_trans_org)
            list_labels.append(torch.tensor(pacs_labels).long())
            list_idx.append( pacs_idx )
            list_size.append(len(pacs_imgs))            
            list_classes.append( len(np.unique(pacs_labels)) )
        
        if self.match_func:
            print('Match Function Updates')
            num_classes= 7
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
        data_imgs_org = torch.cat(list_img_org)
        data_labels = torch.cat(list_labels)
        data_indices = np.array(list_idx)
        data_indices= np.hstack(data_indices)
        self.training_list_size = list_size            

        #No ground truth objects in PACS, for reference we set them same as data indices
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
        data_imgs_org = data_imgs_org[inds]
        data_labels = data_labels[inds]
        data_domains = data_domains[inds].long()
        data_indices = data_indices[inds]
        data_objects = data_objects[inds]

        # Convert to onehot
        out_classes= list_classes[0]
        y = torch.eye(out_classes)
        data_labels = y[data_labels]

        # Convert to onehot
        d = torch.eye(len(self.list_domains))
        data_domains = d[data_domains]
        
        # If shape (B,H,W) change it to (B,C,H,W) with C=1
        if len(data_imgs.shape)==3:
            data_imgs= data_imgs.unsqueeze(1)            
            
        print('Shape: Data ', data_imgs.shape, ' Data w/o augmentation ', data_imgs_org.shape, ' Labels ', data_labels.shape, ' Domains ', data_domains.shape, ' Indices ', data_indices.shape, ' Objects ', data_objects.shape)
        return data_imgs, data_imgs_org, data_labels, data_domains, data_indices, data_objects            
