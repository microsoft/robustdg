import torch
import torch.utils.data as data_utils

#Sklearn
from sklearn.manifold import TSNE

#Pytorch
import torch
from torch.autograd import grad
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import torch.utils.data as data_utils

# Slab Dataset: Flatten the tensor along batch and domain axis
# Input of the shape (Batch, Domain, Feat)
def slab_batch_process(x, y, d, o):
    if len(x.shape) > 2:
        x= x.flatten(start_dim=0, end_dim=1)

    if len(y.shape) > 1:
        y= y.flatten(start_dim=0, end_dim=1)

    if len(d.shape) > 1:
        d= d.flatten(start_dim=0, end_dim=1)

    if len(o.shape) > 1:
        o= o.flatten(start_dim=0, end_dim=1)
    
    return x, y, d, o

def t_sne_plot(X):
    X= X.detach().cpu().numpy()
    X= TSNE(n_components=2).fit_transform(X)
    return X
 
def classifier(x_e, phi, w):
    return torch.matmul(phi(x_e), w)

def erm_loss(temp_logits, target_label):
    loss= F.cross_entropy(temp_logits, target_label.long()).to(cuda)
    return loss

def compute_irm_penalty( logits, target_label, cuda):
    labels= target_label
    scale = torch.tensor(1.).to(cuda).requires_grad_()
    loss = F.cross_entropy(logits*scale, labels.long()).to(cuda)
    g = grad(loss, [scale], create_graph=True)[0].to(cuda)
    # Since g is scalar output, do we need torch.sum?
    ret= torch.sum(g**2)
    return ret 

def cosine_similarity( x1, x2 ):
    cos= torch.nn.CosineSimilarity(dim=1, eps=1e-08)
    return 1.0 - cos(x1, x2)

def l1_dist(x1, x2):
    
    #Broadcasting
    if len(x1.shape) == len(x2.shape) - 1:
        x1=x1.unsqueeze(1)
    if len(x2.shape) == len(x1.shape) - 1:
        x2=x2.unsqueeze(1)
    
    if len(x1.shape) == 3 and len(x2.shape) ==3:
        # Tensor shapes: (N,1,D) and (N,K,D) so x1-x2 would result in (N,K,D)
        return torch.sum( torch.sum(torch.abs(x1 - x2), dim=2) , dim=1 )        
    elif len(x1.shape) ==2 and len(x2.shape) ==2:
        return torch.sum( torch.abs(x1 - x2), dim=1 )
    elif len(x1.shape) ==1 and len(x2.shape) ==1:
        return torch.sum( torch.abs(x1 - x2), dim=0 )
    else:
        print('Error: Expect 1, 2 or 3 rank tensors to compute L1 Norm')
        return

def l2_dist(x1, x2):
    
    #Broadcasting
    if len(x1.shape) == len(x2.shape) - 1:
        x1=x1.unsqueeze(1)
    if len(x2.shape) == len(x1.shape) - 1:
        x2=x2.unsqueeze(1)
    
    if len(x1.shape) == 3 and len(x2.shape) ==3:
        # Tensor shapes: (N,1,D) and (N,K,D) so x1-x2 would result in (N,K,D)
        return torch.sum( torch.sum((x1 - x2)**2, dim=2) , dim=1 )        
    elif len(x1.shape) ==2 and len(x2.shape) ==2:
        return torch.sum( (x1 - x2)**2, dim=1 )
    elif len(x1.shape) ==1 and len(x2.shape) ==1:
        return torch.sum( (x1 - x2)**2, dim=0 )
    else:
        print('Error: Expect 1, 2 or 3 rank tensors to compute L2 Norm')
        return
    
def embedding_dist(x1, x2, pos_metric, tau=0.05, xent=False):
    
    if xent:
        #X1 denotes the batch of anchors while X2 denotes all the negative matches
        #Broadcasting to compute loss for each anchor over all the negative matches
        
        #Only implemnted if x1, x2 are 2 rank tensors
        if len(x1.shape) != 2 or len(x2.shape) !=2:
            print('Error: both should be rank 2 tensors for NT-Xent loss computation')
        
        #Normalizing each vector
        ## Take care to reshape the norm: For a (N*D) vector; the norm would be (N) which needs to be shaped to (N,1) to ensure row wise l2 normalization takes place
        if torch.sum( torch.isnan( x1 ) ):
            print('X1 is nan')
            sys.exit()
            
        if torch.sum( torch.isnan( x2 ) ):
            print('X1 is nan')
            sys.exit()
        
        eps=1e-8
        
        norm= x1.norm(dim=1)
        norm= norm.view(norm.shape[0], 1)
        temp= eps*torch.ones_like(norm)
        
        x1= x1/torch.max(norm, temp)

        if torch.sum( torch.isnan( x1 ) ):
            print('X1 Norm is nan')
            sys.exit()
        
        norm= x2.norm(dim=1)
        norm= norm.view(norm.shape[0], 1)
        temp= eps*torch.ones_like(norm)
        
        x2= x2/torch.max(norm, temp)
        
        if torch.sum( torch.isnan( x2 ) ):
            print('Norm: ', norm, x2 )
            print('X2 Norm is nan')
            sys.exit()
        
        
        # Boradcasting the anchors vector to compute loss over all negative matches
        x1=x1.unsqueeze(1)
        cos_sim= torch.sum( x1*x2, dim=2)
        cos_sim= cos_sim / tau
        
        if torch.sum( torch.isnan( cos_sim ) ):
            print('Cos is nan')
            sys.exit()
                
        loss= torch.sum( torch.exp(cos_sim), dim=1)
        
        if torch.sum( torch.isnan( loss ) ):
            print('Loss is nan')
            sys.exit()
        
        return loss
        
    else:    
        if pos_metric == 'l1':
            return l1_dist(x1, x2)
        elif pos_metric == 'l2':        
            return l2_dist(x1, x2)
        elif pos_metric == 'cos':
            return cosine_similarity( x1, x2 )
    
    
def get_dataloader(args, run, domains, data_case, eval_case, kwargs):
    
    dataset={}

    if args.dataset_name == 'rot_mnist' or args.dataset_name == 'fashion_mnist':        
        if eval_case:
            if args.test_metric in ['match_score'] and args.match_func_aug_case:
                print('Match Function evaluation on self augmentations')
                from data.mnist_loader_match_eval import MnistRotatedAugEval as MnistRotated
            else:
                from data.mnist_loader import MnistRotated
        else:
            from data.mnist_loader import MnistRotated

            
    if args.dataset_name == 'rot_mnist_spur':        
        if eval_case:
            if args.test_metric in ['match_score'] and args.match_func_aug_case:
                print('Match Function evaluation on self augmentations')
                from data.mnist_loader_match_eval_spur import MnistRotatedAugEval as MnistRotated
            else:
                from data.mnist_loader_spur import MnistRotated
        else:
            from data.mnist_loader_spur import MnistRotated
            
    elif args.dataset_name == 'chestxray':
        if eval_case:
            if args.test_metric in ['match_score'] and args.match_func_aug_case:
                print('Match Function evaluation on self augmentations')
                from data.chestxray_loader_match_eval import ChestXRayAugEval as ChestXRay
            else:
                from data.chestxray_loader import ChestXRay
        else:            
            if args.method_name == 'hybrid' and data_case == 'train':            
                print('Hybrid approach with self augmentations')
                from data.chestxray_loader_aug import ChestXRayAug as ChestXRay
            else:
                from data.chestxray_loader import ChestXRay
                
    elif args.dataset_name == 'chestxray_spur':
        if eval_case:
            if args.test_metric in ['match_score'] and args.match_func_aug_case:
                print('Match Function evaluation on self augmentations')
                from data.chestxray_loader_match_eval_spur import ChestXRayAugEval as ChestXRay
            else:
                from data.chestxray_loader_spur import ChestXRay
        else:            
            if args.method_name == 'hybrid' and data_case == 'train':            
                print('Hybrid approach with self augmentations')
                from data.chestxray_loader_aug_spur import ChestXRayAug as ChestXRay
            else:
                from data.chestxray_loader_spur import ChestXRay
                
    elif args.dataset_name == 'pacs':
        if eval_case:
            if args.test_metric in ['match_score'] and args.match_func_aug_case:
                print('Match Function evaluation on self augmentations')
                from data.pacs_loader_match_eval import PACSAugEval as PACS                
            else:
                from data.pacs_loader import PACS
        else:
            if args.method_name == 'hybrid' and data_case == 'train':            
                print('Hybrid approach with self augmentations')
                from data.pacs_loader_aug import PACSAug as PACS
            else:
                from data.pacs_loader import PACS
    
    elif args.dataset_name == 'slab':
        if eval_case and args.test_metric in ['attribute_attack']:   
            from data.slab_loader_spur import SlabData
        else:
            from data.slab_loader import SlabData        

    elif args.dataset_name == 'slab_spur':
        from data.slab_loader_spur import SlabData
        
    if data_case == 'train':
        match_func=True
        batch_size= args.batch_size
    else:
        match_func=False            
        # Can select a higher batch size for val and test domains
        ## TODO: If condition for test batch size less than total size
        
        #Don't try higher batch size in the case of dp-noise trained models to avoid CUDA errors
        if args.dp_noise:
            batch_size= args.batch_size*5
        else:
            batch_size= 512
    
    # Set match_func to True in case of test metric as match_score
    try:
        if args.test_metric in ['match_score', 'feat_eval']:
            match_func=True
    except AttributeError:
        match_func= match_func
            
    try:
        if args.test_metric in ['logit_hist']:
            batch_size=1
    except AttributeError:
        batch_size= batch_size
    
    if args.dataset_name in ['slab', 'slab_spur']:        
        mask_linear=0 
        if args.method_name == 'mask_linear':
            mask_linear= 1            
            if eval_case and args.test_metric in ['attribute_attack']:
                mask_linear= 0
        
        data_obj= SlabData(args, domains, '/slab/', data_case=data_case, match_func=match_func, base_size=args.slab_num_samples, freq_ratio=50, data_dim=args.slab_data_dim, total_slabs=args.slab_total_slabs, seed=run, mask_linear=mask_linear)
        
    elif args.dataset_name in ['pacs', 'vlcs']:
        data_obj= PACS(args, domains, '/pacs/train_val_splits/', data_case=data_case, match_func=match_func)
    
    elif args.dataset_name in ['chestxray']:
        data_obj= ChestXRay(args, domains, '/chestxray/', data_case=data_case, match_func=match_func)

    elif args.dataset_name in ['chestxray_spur']:
        data_obj= ChestXRay(args, domains, '/chestxray_spur/', data_case=data_case, match_func=match_func)
        
    elif args.dataset_name in ['rot_mnist', 'fashion_mnist', 'rot_mnist_spur']:       
        if data_case == 'test' and args.mnist_case not in ['lenet']:
            # Actually by default the seeds 0, 1, 2 are for training and seed 9 is for test; mention that properly in comments
            mnist_subset= 9
        else:
            mnist_subset= run            
                
        print('MNIST Subset: ', mnist_subset)
        data_obj=  MnistRotated(args, domains, mnist_subset, '/mnist/', data_case=data_case, match_func=match_func)
        
    dataset['data_loader']= data_utils.DataLoader(data_obj, batch_size=batch_size, shuffle=True, **kwargs )
    
    dataset['data_obj']= data_obj
    dataset['total_domains']= len(domains)
    dataset['domain_list']= domains
    dataset['base_domain_size']= data_obj.base_domain_size       
    dataset['domain_size_list']= data_obj.training_list_size    
    
    print(data_case, data_obj.base_domain_size, data_obj.training_list_size)
    
    if eval_case and args.test_metric in ['match_score'] and args.match_func_aug_case:
        dataset['total_domains']= 2
        dataset['domain_list']= ['aug', 'org']
    
    return dataset
