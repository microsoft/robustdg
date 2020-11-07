#Common imports
import os
import random
import copy
import numpy as np

#Sklearn
from scipy.stats import bernoulli

#Pytorch
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms

#Pillow
from PIL import Image, ImageColor, ImageOps 

#Base Class
from .data_loader import BaseDataLoader

class MnistRotatedAugEval(BaseDataLoader):
    def __init__(self, args, list_train_domains, mnist_subset, root, transform=None, data_case='train', match_func=False, download=True):
        
        super().__init__(args, list_train_domains, root, transform, data_case, match_func) 
        self.mnist_subset = mnist_subset
        self.download = download
        
        self.train_data, self.train_labels, self.train_domain, self.train_indices, self.train_spur = self._get_data()

    def load_inds(self):
        data_dir= self.root + self.args.dataset_name + '_' + self.args.model_name + '_indices'
        if self.data_case != 'val':
            return np.load(data_dir + '/supervised_inds_' + str(self.mnist_subset) + '.npy')
        else:
            return np.load(data_dir + '/val' + '/supervised_inds_' + str(self.mnist_subset) + '.npy')
            
    def _get_data(self):
                
        if self.args.dataset_name =='rot_mnist_spur':
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
            
        elif self.args.dataset_name == 'fashion_mnist_spur':
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
        print(type(mnist_imgs), mnist_labels.shape, mnist_imgs.shape)

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
        
        color_list=['red', 'blue', 'green', 'orange', 'yellow', 'brown', 'pink', 'magenta', 'olive', 'cyan']

        # Choose subsets that should be included into the training
        training_list_img = {'aug':[], 'org':[] }
        training_list_labels = {'aug':[], 'org':[] }
        training_list_idx= {'aug':[], 'org':[] }
        training_list_spur= {'aug':[], 'org':[]}
        training_list_size= {'aug':0, 'org':0 }
        training_out_classes={'aug':[], 'org':[] }
        
        # key: class labels, val: data indices
        indices_dict={}
        for i in range(len(mnist_imgs)):
            key= int( mnist_labels[i].numpy() )
            if key not in indices_dict.keys():
                indices_dict[key]=[]
            indices_dict[key].append( i )

            
        image_counter=0
        for domain in self.list_train_domains:
            
            # Adding color with 70 percent probability
            rand_var= bernoulli.rvs(0.7, size=mnist_size)
            
            # Run transforms
            mnist_img_rot= torch.zeros((mnist_size, self.args.img_c, self.args.img_w, self.args.img_h))
            mnist_img_rot_org= torch.zeros((mnist_size, self.args.img_c, self.args.img_w, self.args.img_h))
            mnist_idx=[]
            
            # Shuffling the images to create random across domains
            curr_indices_dict= copy.deepcopy( indices_dict )
            for key in curr_indices_dict.keys():
                random.shuffle( curr_indices_dict[key] )
            
            for i in range(len(mnist_imgs)):
                
                curr_image= to_pil(mnist_imgs[i])
                #Color the image 
                if rand_var[i]:
                    # Change colors per label for test domains relative to the train domains
                    if self.data_case == 'test':
#                         curr_image = ImageOps.colorize(curr_image, black ="black", white =color_list[(mnist_labels[i].item()+1)%10]  )
                        curr_image = ImageOps.colorize(curr_image, black ="black", white =color_list[mnist_labels[i].item()])    
                    else:
                        curr_image = ImageOps.colorize(curr_image, black ="black", white =color_list[mnist_labels[i].item()])    
                else:
                    curr_image = ImageOps.colorize(curr_image, black ="black", white ="white")                 
                    
                if domain == '0':
                    mnist_img_rot[i]= to_tensor( to_augment(curr_image) )
                    mnist_img_rot_org[i]= to_tensor(curr_image)
                else:
                    mnist_img_rot[i]= to_tensor( to_augment( transforms.functional.rotate( curr_image, int(domain) ) ) )        
                    mnist_img_rot_org[i]= to_tensor( transforms.functional.rotate( curr_image, int(domain) ) )                        
                    
                mnist_idx.append( image_counter )
                image_counter+= 1                
            
            print('Source Domain ', domain)
            training_list_img['aug'].append(mnist_img_rot)            
            training_list_img['org'].append(mnist_img_rot_org)      
            
            
            training_list_labels['aug'].append(mnist_labels)
            training_list_labels['org'].append(mnist_labels)
            
            training_list_idx['aug'].append( mnist_idx )            
            training_list_idx['org'].append( mnist_idx )            
                        
            training_list_spur['aug'].append( rand_var )            
            training_list_spur['org'].append( rand_var )            
                        
            training_list_size['aug']+= mnist_img_rot.shape[0]
            training_list_size['org']+= mnist_img_rot.shape[0]    
            
        if self.match_func:
            print('Match Function Updates')
            num_classes= 10
            for y_c in range(num_classes):
                for key in ['aug', 'org']:
                    base_class_size=0
                    base_class_idx=-1
                    
                    curr_class_size=0                    
                    for d_idx, domain in enumerate( self.list_train_domains ):
                        class_idx= training_list_labels[key][d_idx] == y_c
                        curr_class_size+= training_list_labels[key][d_idx][class_idx].shape[0]
                        
                    if base_class_size < curr_class_size:
                        base_class_size= curr_class_size
                        if key == 'aug':
                            base_class_idx= 0
                        else:
                            base_class_idx= 1                            
                        
                self.base_domain_size += base_class_size
                print('Max Class Size: ', base_class_size, base_class_idx, y_c )
                   
        # Stack
        train_imgs = torch.cat(training_list_img['aug'] + training_list_img['org'] )
        train_labels = torch.cat(training_list_labels['aug'] + training_list_labels['org'] )
        train_indices = np.array(training_list_idx['aug']+training_list_idx['org']) 
        train_indices= np.hstack(train_indices)
        print('Train indices', len(np.unique(train_indices)) )
        
        train_spur =  np.array(training_list_spur['aug']+training_list_spur['org']) 
        train_spur= np.hstack(train_spur)            
                
        training_out_classes= training_out_classes['aug'] + training_out_classes['org']
        self.training_list_size = [ training_list_size['aug'],  training_list_size['org'] ]           
   
        print(train_imgs.shape, train_labels.shape, train_indices.shape)
        print(self.training_list_size, self.base_domain_size)        
        print(training_out_classes)
        
        # Create domain labels
        train_domains = torch.zeros(train_labels.size())
        domain_start=0
        for idx in range(len(self.training_list_size)):
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
        train_spur = train_spur[inds]

        # Convert to onehot
        y = torch.eye(10)
        train_labels = y[train_labels]

        # Convert to onehot
        d = torch.eye(len(self.training_list_size))
        train_domains = d[train_domains]
        
        print(train_imgs.shape, train_labels.shape, train_domains.shape, train_indices.shape, train_spur.shape)
        return train_imgs.unsqueeze(1), train_labels, train_domains, train_indices, train_spur
