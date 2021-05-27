#Common imports
import numpy as np
import sys
import os
import random
import copy

#Sklearn
from scipy.stats import bernoulli

#Pillow
from PIL import Image, ImageColor, ImageOps 

#Pytorch
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms

def generate_rotated_domain_data(imgs, labels, data_case, dataset, indices, domain, save_dir, img_w, img_h):    

    # Get total number of labeled examples
    mnist_labels = labels[indices]
    mnist_imgs = imgs[indices]
    mnist_size = mnist_labels.shape[0] 

    to_pil=  transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_w, img_h))
        ])

    to_augment= transforms.Compose([
            transforms.RandomResizedCrop(img_w, scale=(0.7,1.0)),
            transforms.RandomHorizontalFlip()
        ])

    to_tensor=  transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    if dataset == 'rot_mnist_spur':
        color_list=['red', 'blue', 'green', 'orange', 'yellow', 'brown', 'pink', 'magenta', 'olive', 'cyan']    
        # Adding color with 70 percent probability
        rand_var= bernoulli.rvs(0.7, size=mnist_size)
    
    # Run transforms
    if dataset == 'rot_mnist_spur':
        mnist_img_rot= torch.zeros((mnist_size, 3, img_w, img_h))        
        mnist_img_rot_org= torch.zeros((mnist_size, 3, img_w, img_h))        
    else:
        mnist_img_rot= torch.zeros((mnist_size, img_w, img_h))
        mnist_img_rot_org= torch.zeros((mnist_size, img_w, img_h))
        
    mnist_idx=[]

    for i in range(len(mnist_imgs)):
        
        curr_image= to_pil(mnist_imgs[i])         
        
        #Color the image
        if dataset == 'rot_mnist_spur':
            if rand_var[i]:
                # Change colors per label for test domains relative to the train domains
                if data_case == 'test':
                    curr_image = ImageOps.colorize(curr_image, black ="black", white =color_list[mnist_labels[i].item()])    
                    # Choose this for test domain with permuted colors
#                     curr_image = ImageOps.colorize(curr_image, black ="black", white =color_list[(mnist_labels[i].item()+1)%10]  )
                else:
                    curr_image = ImageOps.colorize(curr_image, black ="black", white =color_list[mnist_labels[i].item()])    
            else:
                curr_image = ImageOps.colorize(curr_image, black ="black", white ="white")               
        
        #Rotation
        if domain == '0':
            img_rotated= curr_image
        else:
            img_rotated= transforms.functional.rotate( curr_image, int(domain) )

        mnist_img_rot_org[i]= to_tensor(img_rotated)        
        #Augmentation
        mnist_img_rot[i]= to_tensor(to_augment(img_rotated))        

    if data_case == 'train':
        torch.save(mnist_img_rot, save_dir + '_data.pt')    
        
    torch.save(mnist_img_rot_org, save_dir + '_org_data.pt')        
    torch.save(mnist_labels, save_dir + '_label.pt')    
    
    if dataset == 'rot_mnist_spur':
        np.save(save_dir + '_spur.npy', rand_var)
        
    print('Data Case: ', data_case, ' Source Domain: ', domain, ' Shape: ', mnist_img_rot.shape, mnist_img_rot_org.shape, mnist_labels.shape)        
    
    return

# Main Function

dataset= sys.argv[1]
model= sys.argv[2]

#Generate Dataset for Rotated / Fashion MNIST
base_dir= 'datasets/mnist/'
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

if model == 'resnet18':
    if dataset == 'rot_mnist' or dataset == 'rot_mnist_spur':
        # Generate 10 random subsets of size 2,000 each for Rotated MNIST 
        data_size=60000
        subset_size=2000
        val_size= 400    
        total_subset=10
        img_w= 224
        img_h= 224

    elif dataset == 'fashion_mnist':
        # Genetate 10 random subsets of size 10,000 each for Fashion MNIST
        data_size=60000
        subset_size=10000
        val_size= 2000
        total_subset=10
        img_w= 224
        img_h= 224        
        
elif model == 'lenet':
    if dataset == 'rot_mnist':
        # Generate 10 random subsets of size 1,000 each for Rotated MNIST 
        data_size=60000
        subset_size=1000
        val_size= 200
        total_subset=10
        img_w= 32
        img_h= 32
    elif dataset == 'fashion_mnist':
        print('Fashion MNIST not implemented for LeNet')
    
elif model == 'domain_bed':
    if dataset == 'rot_mnist':
        # Generate 10 random subsets of size 0.8*70,000 each for Rotated MNIST 
        data_size=70000
        subset_size=55000
        val_size= 1000
        total_subset=10
        img_w= 28
        img_h= 28
    elif dataset == 'fashion_mnist':
        print('Fashion MNIST not implemented for DomainBed')    
        
data_dir= base_dir + dataset + '_' + model + '/'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
    
    
if dataset =='rot_mnist' or dataset == 'rot_mnist_spur':
    data_obj_train= datasets.MNIST(base_dir,
                                train=True,
                                download=True,
                                transform=transforms.ToTensor()
                            )

    data_obj_test= datasets.MNIST(base_dir,
                                train=False,
                                download=True,
                                transform=transforms.ToTensor()
                            )
    mnist_imgs= torch.cat((data_obj_train.data, data_obj_test.data))
    mnist_labels= torch.cat((data_obj_train.targets, data_obj_test.targets))

elif dataset == 'fashion_mnist':
    data_obj_train= datasets.FashionMNIST(base_dir,
                                        train=True,
                                        download=True,
                                        transform=transforms.ToTensor()
                                    )

    data_obj_test= datasets.FashionMNIST(base_dir,
                                train=False,
                                download=True,
                                transform=transforms.ToTensor()
                            )
    mnist_imgs= torch.cat((data_obj_train.data, data_obj_test.data))
    mnist_labels= torch.cat((data_obj_train.targets, data_obj_test.targets))
    
    
# For testing over different base objects; seed 9
seed_list= [0, 1, 2, 9]    
domains= [0, 15, 30, 45, 60, 75, 90]

for seed in seed_list:
    
    # Random Seed
    np.random.seed(seed*10)     
    # Indices
    res=np.random.choice(data_size, subset_size+val_size)
    print('Seed: ', seed)
    for domain in domains:
        
        #Train
        data_case= 'train'
        if not os.path.exists(data_dir + data_case +  '/'):
            os.makedirs(data_dir + data_case + '/')

        save_dir= data_dir + data_case + '/' + 'seed_' + str(seed) + '_domain_' + str(domain)
        indices= res[:subset_size]
        generate_rotated_domain_data(mnist_imgs, mnist_labels, data_case, dataset, indices, domain, save_dir, img_w, img_h)     
        
        #Test
        data_case= 'test'
        if not os.path.exists(data_dir +  data_case  +  '/'):
            os.makedirs(data_dir + data_case + '/')
            
        save_dir= data_dir + data_case + '/' + 'seed_' + str(seed) + '_domain_' + str(domain)
        indices= res[:subset_size]
        generate_rotated_domain_data(mnist_imgs, mnist_labels, data_case, dataset, indices, domain, save_dir, img_w, img_h)             
        
        #Val 
        data_case= 'val'
        if not os.path.exists(data_dir +  data_case +  '/'):
            os.makedirs(data_dir + data_case + '/')
        
        save_dir= data_dir + data_case +  '/' + 'seed_' + str(seed) + '_domain_' + str(domain)
        indices= res[subset_size:]
        generate_rotated_domain_data(mnist_imgs, mnist_labels, data_case, dataset, indices, domain, save_dir, img_w, img_h)
