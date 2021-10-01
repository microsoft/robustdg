#Common imports
import numpy as np
import sys
import os
import argparse
import random
import copy
import os 

#Sklearn
from scipy.stats import bernoulli

#Pillow
from PIL import Image, ImageColor, ImageOps 

#Pytorch
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms

def generate_rotated_domain_data(imgs, labels, data_case, dataset, indices, domain, save_dir, img_w, img_h, cmnist_permute):    

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
                    if cmnist_permute:
                        # Choose this for test domain with permuted colors
                        curr_image = ImageOps.colorize(curr_image, black ="black", white =color_list[(mnist_labels[i].item()+1)%10] )
                    else:
                        curr_image = ImageOps.colorize(curr_image, black ="black", white =color_list[mnist_labels[i].item()])    
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

    if data_case == 'train' or data_case == 'val':
        torch.save(mnist_img_rot, save_dir + '_data.pt')    
        
    torch.save(mnist_img_rot_org, save_dir + '_org_data.pt')        
    torch.save(mnist_labels, save_dir + '_label.pt')    
    
    if dataset == 'rot_mnist_spur':
        np.save(save_dir + '_spur.npy', rand_var)
    
    print('Data Case: ', data_case, ' Source Domain: ', domain, ' Shape: ', mnist_img_rot.shape, mnist_img_rot_org.shape, mnist_labels.shape)        
    
    return

# Main Function

# Input Parsing
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='rot_mnist', 
                    help='Datasets: rot_mnist; fashion_mnist; rot_mnist_spur')
parser.add_argument('--model', type=str, default='resnet18', 
                    help='Base Models: resnet18; lenet')
parser.add_argument('--data_size', type=int, default=60000)
parser.add_argument('--subset_size', type=int, default=2000)
parser.add_argument('--img_w', type=int, default=224)
parser.add_argument('--img_h', type=int, default=224)
parser.add_argument('--cmnist_permute', type=int, default=0)

args = parser.parse_args()

dataset= args.dataset
model= args.model
img_w= args.img_w
img_h= args.img_h
data_size= args.data_size
subset_size= args.subset_size
val_size= int(args.subset_size/5)
cmnist_permute= args.cmnist_permute

#Generate Dataset for Rotated / Fashion MNIST
#TODO: Manage OS Env from args
os_env=0
if os_env:
    base_dir= os.getenv('PT_DATA_DIR') + '/mnist/'
else:
    base_dir= 'data/datasets/mnist/'
    
if not os.path.exists(base_dir):
    os.makedirs(base_dir)
        
        
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
# Seed 9 only for test data, See 0:3 for train data
seed_list= [0, 1, 2, 9]    
domains= [0, 15, 30, 45, 60, 75, 90]

for seed in seed_list:
    
    # Random Seed
    random.seed(seed*10)
    np.random.seed(seed*10)     
    torch.manual_seed(seed*10)    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed*10)
        
    # Indices
    res=np.random.choice(data_size, subset_size+val_size)
    print('Seed: ', seed)
    for domain in domains:        
        
        # The case of permuted test domain for colored rotated MNIST, only update test data
        if dataset == 'rot_mnist_spur' and cmnist_permute:            
            #Test
            data_case= 'test'            
            save_dir= data_dir + data_case + '/' + 'seed_' + str(seed) + '_domain_' + str(domain)
            indices= res[:subset_size]        
            if seed in [9] and domain in [0, 15, 30, 45, 60, 75, 90]:
                generate_rotated_domain_data(mnist_imgs, mnist_labels, data_case, dataset, indices, domain, save_dir, img_w, img_h, cmnist_permute)         
            
            continue
            
        #Train
        data_case= 'train'
        if not os.path.exists(data_dir + data_case +  '/'):
            os.makedirs(data_dir + data_case + '/')

        save_dir= data_dir + data_case + '/' + 'seed_' + str(seed) + '_domain_' + str(domain)
        indices= res[:subset_size]      

        if model == 'resnet18':
            if seed in [0, 1, 2] and domain in [15, 30, 45, 60, 75]:
                generate_rotated_domain_data(mnist_imgs, mnist_labels, data_case, dataset, indices, domain, save_dir, img_w, img_h, cmnist_permute)                   
        elif model in ['lenet']:
            if seed in [0, 1, 2] and domain in [0, 15, 30, 45, 60, 75]:
                generate_rotated_domain_data(mnist_imgs, mnist_labels, data_case, dataset, indices, domain, save_dir, img_w, img_h, cmnist_permute)                   
                    
        #Val 
        data_case= 'val'
        if not os.path.exists(data_dir +  data_case +  '/'):
            os.makedirs(data_dir + data_case + '/')
        
        save_dir= data_dir + data_case +  '/' + 'seed_' + str(seed) + '_domain_' + str(domain)
        indices= res[subset_size:]
        
        if model == 'resnet18':
            if seed in [0, 1, 2] and domain in [15, 30, 45, 60, 75]:
                generate_rotated_domain_data(mnist_imgs, mnist_labels, data_case, dataset, indices, domain, save_dir, img_w, img_h, cmnist_permute)                
        elif model in ['lenet']:
            if seed in [0, 1, 2] and domain in [0, 15, 30, 45, 60, 75]:
                generate_rotated_domain_data(mnist_imgs, mnist_labels, data_case, dataset, indices, domain, save_dir, img_w, img_h, cmnist_permute)                
            
        #Test
        data_case= 'test'
        if not os.path.exists(data_dir +  data_case  +  '/'):
            os.makedirs(data_dir + data_case + '/')
            
        save_dir= data_dir + data_case + '/' + 'seed_' + str(seed) + '_domain_' + str(domain)
        indices= res[:subset_size]
                
        if model == 'resnet18':
            if seed in [0, 1, 2, 9] and domain in [0, 90]:
                generate_rotated_domain_data(mnist_imgs, mnist_labels, data_case, dataset, indices, domain, save_dir, img_w, img_h, cmnist_permute)             
        elif model in ['lenet', 'lenet_mdg']:
            if seed in [0, 1, 2] and domain in [0, 15, 30, 45, 60, 75]:
                generate_rotated_domain_data(mnist_imgs, mnist_labels, data_case, dataset, indices, domain, save_dir, img_w, img_h, cmnist_permute)

                
        # Extra data sampling for carrying out the attribute attack on spurious rotated mnist
        if dataset == 'rot_mnist_spur':
            
            #Train
            data_case= 'train'
            save_dir= data_dir + data_case + '/' + 'seed_' + str(seed) + '_domain_' + str(domain)
            indices= res[:subset_size]      
            if seed in [0, 1, 2] and domain in [0, 90]:
                generate_rotated_domain_data(mnist_imgs, mnist_labels, data_case, dataset, indices, domain, save_dir, img_w, img_h, cmnist_permute)                   
                    
            #Val 
            data_case= 'val'
            save_dir= data_dir + data_case +  '/' + 'seed_' + str(seed) + '_domain_' + str(domain)
            indices= res[subset_size:]        
            if seed in [0, 1, 2] and domain in [0, 90]:
                generate_rotated_domain_data(mnist_imgs, mnist_labels, data_case, dataset, indices, domain, save_dir, img_w, img_h, cmnist_permute)                
            
            #Test
            data_case= 'test'            
            save_dir= data_dir + data_case + '/' + 'seed_' + str(seed) + '_domain_' + str(domain)
            indices= res[:subset_size]        
            if seed in [9] and domain in [15, 30, 45, 60, 75]:
                generate_rotated_domain_data(mnist_imgs, mnist_labels, data_case, dataset, indices, domain, save_dir, img_w, img_h, cmnist_permute)             

            
            