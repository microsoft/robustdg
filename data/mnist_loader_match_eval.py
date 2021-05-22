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

class MnistRotatedAugEval(BaseDataLoader):
    def __init__(self, args, list_domains, mnist_subset, root, transform=None, data_case='train', match_func=False, download=True):
        
        super().__init__(args, list_domains, root, transform, data_case, match_func) 
        self.mnist_subset = mnist_subset
        self.download = download
        
        self.data, self.labels, self.domains, self.indices, self.objects = self._get_data()

    def load_inds(self):
        data_dir= self.root + self.args.dataset_name + '_' + self.args.model_name + '_indices'
        if self.data_case != 'val':
            return np.load(data_dir + '/supervised_inds_' + str(self.mnist_subset) + '.npy')
        else:
            return np.load(data_dir + '/val' + '/supervised_inds_' + str(self.mnist_subset) + '.npy')
            
    def _get_data(self):
                
        if self.args.dataset_name =='rot_mnist':
            data_obj_train= datasets.MNIST(self.root,
                                        train=True,
                                        download=self.download,
                                        transform=transforms.ToTensor()
                                    )
            
            data_obj_test= datasets.MNIST(self.root,
                                        train=False,
                                        download=self.download,
                                        transform=transforms.ToTensor()
                                    )
            mnist_imgs= torch.cat((data_obj_train.data, data_obj_test.data))
            mnist_labels= torch.cat((data_obj_train.targets, data_obj_test.targets))
            
        elif self.args.dataset_name == 'fashion_mnist':
            data_obj_train= datasets.FashionMNIST(self.root,
                                                train=True,
                                                download=self.download,
                                                transform=transforms.ToTensor()
                                            )
            
            data_obj_test= datasets.FashionMNIST(self.root,
                                        train=False,
                                        download=self.download,
                                        transform=transforms.ToTensor()
                                    )
            mnist_imgs= torch.cat((data_obj_train.data, data_obj_test.data))
            mnist_labels= torch.cat((data_obj_train.targets, data_obj_test.targets))
            
        # Get total number of labeled examples
        sup_inds = self.load_inds()
        mnist_labels = mnist_labels[sup_inds]
        mnist_imgs = mnist_imgs[sup_inds]
        mnist_size = mnist_labels.shape[0] 

        to_pil=  transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((self.args.img_w, self.args.img_h))
            ])
        
        to_augment= transforms.Compose([
                transforms.RandomResizedCrop(self.args.img_w, scale=(0.7,1.0)),
                transforms.RandomHorizontalFlip(),            
            ])
        
        to_tensor=  transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])

        # Choose subsets that should be included into the training
        list_img = {'aug':[], 'org':[] }
        list_labels = {'aug':[], 'org':[] }
        list_idx= {'aug':[], 'org':[] }
        list_size= {'aug':0, 'org':0 }
        list_classes={'aug':[], 'org':[] }

            
        image_counter=0
        for domain in self.list_domains:
            # Run transforms
            mnist_img_rot= torch.zeros((mnist_size, self.args.img_w, self.args.img_h))
            mnist_img_rot_org= torch.zeros((mnist_size, self.args.img_w, self.args.img_h))
            mnist_idx=[]
            
            for i in range(len(mnist_imgs)):
                if domain == '0':
                    mnist_img_rot[i]= to_tensor( to_augment( to_pil(mnist_imgs[i]) ) )
                    mnist_img_rot_org[i]= to_tensor(to_pil(mnist_imgs[i]))
                else:
                    mnist_img_rot[i]= to_tensor( to_augment( transforms.functional.rotate( to_pil(mnist_imgs[i]), int(domain) ) ) )        
                    mnist_img_rot_org[i]= to_tensor( transforms.functional.rotate( to_pil(mnist_imgs[i]), int(domain) ) )        
                    
                mnist_idx.append( image_counter )
                image_counter+= 1                
            
            print('Source Domain ', domain)
            list_img['aug'].append(mnist_img_rot)            
            list_img['org'].append(mnist_img_rot_org)      
                        
            list_labels['aug'].append(mnist_labels)
            list_labels['org'].append(mnist_labels)
            
            list_idx['aug'].append( mnist_idx )            
            list_idx['org'].append( mnist_idx )            
            
            list_size['aug']+= mnist_img_rot.shape[0]
            list_size['org']+= mnist_img_rot.shape[0]    
            
        if self.match_func:
            print('Match Function Updates')
            num_classes= 10
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
        self.training_list_size = [ list_size['aug'],  list_size['org'] ]           
           
        #Rotated MNIST the objects are same the data indices
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
        y = torch.eye(10)
        data_labels = y[data_labels]

        # Convert to onehot
        d = torch.eye(len(self.training_list_size))
        data_domains = d[data_domains]
        
        # If shape (B,H,W) change it to (B,C,H,W) with C=1
        if len(data_imgs.shape)==3:
            data_imgs= data_imgs.unsqueeze(1)        
        
        print('Shape: Data ', data_imgs.shape, ' Labels ', data_labels.shape, ' Domains ', data_domains.shape, ' Indices ', data_indices.shape, ' Objects ', data_objects.shape)
        return data_imgs, data_labels, data_domains, data_indices, data_objects