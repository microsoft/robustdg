"""Pytorch Dataset object that loads MNIST and SVHN. It returns x,y,s where s=0 when x,y is taken from MNIST."""

import os
import random
import copy
import numpy as np
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms

class MnistRotated(data_utils.Dataset):
    def __init__(self, dataset_name, list_train_domains, mnist_subset, root, transform=None, data_case='train', download=True):
        self.dataset_name= dataset_name
        self.list_train_domains = list_train_domains
        self.mnist_subset = mnist_subset
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.data_case = data_case
        self.download = download
        self.base_domain_idx= -1
        self.base_domain_size= 0
        self.training_list_size=[]

        self.train_data, self.train_labels, self.train_domain, self.train_indices = self._get_data()

    def load_inds(self):
        if self.mnist_subset == -1:
            # Select a random batch of size 2000 from the total 60,000 samples of MNIST
            if self.dataset_name == 'rot_mnist':
                if self.data_case != 'val':
                    res= np.random.choice(60000, 1000)
                else:
                    res= np.random.choice(60000, 100)
            return res
#             res=[]
#             for subset in range(10):
#                 temp= np.load(self.root + '/supervised_inds_' + str(subset) + '.npy' )
#                 res.append(temp)
#             res= np.array(res)
#             res= np.reshape( res, (res.shape[0]*res.shape[1]) )
#             print(res.shape)
#             return res
        else:
            if self.dataset_name == 'rot_mnist':
                data_dir= self.root + '/rot_mnist_lenet_indices'
#                 data_dir= self.root + '/rot_mnist_indices'
                
            if self.data_case != 'val':
                return np.load(data_dir + '/supervised_inds_' + str(self.mnist_subset) + '.npy')
            else:
                return np.load(data_dir + '/val' + '/supervised_inds_' + str(self.mnist_subset) + '.npy')
            
    def _get_data(self):
        
        if self.dataset_name =='rot_mnist':
            data_obj= datasets.MNIST(self.root,
                                        train=True,
                                        download=self.download,
                                        transform=transforms.ToTensor()
                                    )
        
        train_loader = torch.utils.data.DataLoader(data_obj,
                                                   batch_size=60000,
                                                   shuffle=False)

        for i, (x, y) in enumerate(train_loader):
            mnist_imgs = x
            mnist_labels = y

        # Get total number of labeled examples
        sup_inds = self.load_inds()
        mnist_labels = mnist_labels[sup_inds]
        mnist_imgs = mnist_imgs[sup_inds]
        mnist_size = mnist_labels.shape[0] 
        print(type(mnist_imgs), mnist_labels.shape, mnist_imgs.shape)

        to_pil=  transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((32, 32))
            ])
        
        to_augment= transforms.Compose([
                transforms.RandomResizedCrop(32, scale=(0.7,1.0)),
                transforms.RandomHorizontalFlip(),            
            ])
        
        to_tensor=  transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])

        # Choose subsets that should be included into the training
        training_list_img = []
        training_list_labels = []
        training_list_idx= []
        training_list_size= []

        # key: class labels, val: data indices
        indices_dict={}
        for i in range(len(mnist_imgs)):
            key= int( mnist_labels[i].numpy() )
            if key not in indices_dict.keys():
                indices_dict[key]=[]
            indices_dict[key].append( i )
        
        for domain in self.list_train_domains:
            # Run transforms
            mnist_img_rot= torch.zeros((mnist_size, 32, 32))
            mnist_idx=[]
            
            # Shuffling the images to create random across domains
            curr_indices_dict= copy.deepcopy( indices_dict )
            for key in curr_indices_dict.keys():
                random.shuffle( curr_indices_dict[key] )
            
            for i in range(len(mnist_imgs)):
                if domain == '0':
                    mnist_img_rot[i]= to_tensor(to_pil(mnist_imgs[i]))
                else:
                    if self.data_case =='train' and self.dataset_name =="fashion_mnist":
                        mnist_img_rot[i]= to_tensor( to_augment( transforms.functional.rotate( to_pil(mnist_imgs[i]), int(domain) ) ) )        
                    else:
                        mnist_img_rot[i]= to_tensor( transforms.functional.rotate( to_pil(mnist_imgs[i]), int(domain) ) )        
                    
                mnist_idx.append( i )
            
            #Sanity Check
            #print('Sanity Check for alloting env indices: ', curr_indices_dict)

            print('Source Domain ', domain)
            training_list_img.append(mnist_img_rot)
            training_list_labels.append(mnist_labels)
            training_list_idx.append(mnist_idx)
            training_list_size.append(mnist_img_rot.shape[0])
             
        # Making domain size equivalent everywhere by random sampling
        if self.data_case == 'train':
            num_classes= 10
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
        train_indices = np.reshape( train_indices, ( train_indices.shape[0]*train_indices.shape[1] ) )    
        self.training_list_size= training_list_size
        print(train_imgs.shape, train_labels.shape, train_indices.shape)
        print(self.training_list_size)
        
        # Create domain labels
        train_domains = torch.zeros(train_labels.size())
        for idx in range(len(self.list_train_domains)):
            train_domains[idx * mnist_size: (idx+1) * mnist_size] += idx

        # Shuffle everything one more time
        inds = np.arange(train_labels.size()[0])
        np.random.shuffle(inds)
        train_imgs = train_imgs[inds]
        train_labels = train_labels[inds]
        train_domains = train_domains[inds].long()
        train_indices = train_indices[inds]

        # Convert to onehot
        y = torch.eye(10)
        train_labels = y[train_labels]

        # Convert to onehot
        d = torch.eye(len(self.list_train_domains))
        train_domains = d[train_domains]
        
        print(train_imgs.shape, train_labels.shape, train_domains.shape, train_indices.shape)
        return train_imgs.unsqueeze(1), train_labels, train_domains, train_indices

    def __len__(self):
        return self.train_labels.shape[0]

    def __getitem__(self, index):
        x = self.train_data[index]
        y = self.train_labels[index]
        d = self.train_domain[index]
        idx = self.train_indices[index]
            
        if self.transform is not None:
            x = self.transform(x)
        return x, y, d, idx

    def get_size(self):
        return self.train_labels.shape[0]

if __name__ == "__main__":
    from torchvision.utils import save_image

    seed = 1

    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

    list_train_domains = ['0', '15', '30', '45', '60']
    num_supervised = 1000

    train_loader = data_utils.DataLoader(
        MnistRotated(list_train_domains, num_supervised, seed, '../dataset/', train=True),
        batch_size=100,
        shuffle=False)

    y_array = np.zeros(10)
    d_array = np.zeros(5)

    for i, (x, y, d) in enumerate(train_loader):
        y_array += y.sum(dim=0).cpu().numpy()
        d_array += d.sum(dim=0).cpu().numpy()

        if i == 0:
            print(y)
            print(d)
            n = min(x.size(0), 8)
            comparison = x[:n].view(-1, 1, 16, 16)
            save_image(comparison.cpu(),
                       'reconstruction_rotation_train.png', nrow=n)

    print(y_array, d_array)

    test_loader = data_utils.DataLoader(
        MnistRotated(list_train_domains, seed, '../dataset/', train=False),
        batch_size=100,
        shuffle=False)

    y_array = np.zeros(10)
    d_array = np.zeros(5)

    for i, (x, y, d) in enumerate(test_loader):
        y_array += y.sum(dim=0).cpu().numpy()
        d_array += d.sum(dim=0).cpu().numpy()

        if i == 0:
            print(y)
            print(d)
            n = min(x.size(0), 8)
            comparison = x[:n].view(-1, 1, 16, 16)
            save_image(comparison.cpu(),
                       'reconstruction_rotation_test.png', nrow=n)

    print(y_array, d_array)