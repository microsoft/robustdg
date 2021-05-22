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
                transforms.RandomHorizontalFlip()
            ])
        
        to_tensor=  transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])

        # Choose subsets that should be included into the training
        list_img = []
        list_labels = []
        list_idx= []
        list_size= []

#         if self.data_case == 'test':
#             self.list_train_domains= ['30', '45']
            
        for domain in self.list_domains:
            # Run transforms
            mnist_img_rot= torch.zeros((mnist_size, self.args.img_w, self.args.img_h))
            mnist_idx=[]
                        
            for i in range(len(mnist_imgs)):
                #Rotation
                if domain == '0':
                    img_rotated= to_pil(mnist_imgs[i])
                else:
                    img_rotated= transforms.functional.rotate( to_pil(mnist_imgs[i]), int(domain) )
                    
                #Augmentation
                if self.data_case =='train' and self.args.dataset_name =="fashion_mnist":
                    mnist_img_rot[i]= to_tensor(to_augment(img_rotated))        
                else:
                    mnist_img_rot[i]= to_tensor(img_rotated)        
                    
                mnist_idx.append( i )
            
            print('Source Domain ', domain)
            list_img.append(mnist_img_rot)
            list_labels.append(mnist_labels)
            list_idx.append(mnist_idx)
            list_size.append(mnist_img_rot.shape[0])
             
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
        
        #Rotated MNIST the objects are same the data indices
        data_objects= copy.deepcopy(data_indices)
        
        # Create domain labels
        data_domains = torch.zeros(data_labels.size())
        for idx in range(len(self.list_domains)):
            data_domains[idx * mnist_size: (idx+1) * mnist_size] += idx

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
        d = torch.eye(len(self.list_domains))
        data_domains = d[data_domains]
        
        # If shape (B,H,W) change it to (B,C,H,W) with C=1
        if len(data_imgs.shape)==3:
            data_imgs= data_imgs.unsqueeze(1)        
        
        print('Shape: Data ', data_imgs.shape, ' Labels ', data_labels.shape, ' Domains ', data_domains.shape, ' Indices ', data_indices.shape, ' Objects ', data_objects.shape)
        return data_imgs, data_labels, data_domains, data_indices, data_objects
