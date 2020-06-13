import numpy as np
import argparse
import copy
import random
import json

import torch
from torch.autograd import grad
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import torch.utils.data as data_utils

from sklearn.manifold import TSNE

def t_sne_plot(X):
#     X= X.view(X.shape[0], X.shape[1]*X.shape[2]*X.shape[3])
    X= X.detach().cpu().numpy()
    X= TSNE(n_components=2).fit_transform(X)
    return X
 
def classifier(x_e, phi, w):
    return torch.matmul(phi(x_e), w)

def erm_loss(temp_logits, target_label):
    #mse= torch.nn.MSELoss(reduction="none")
    #print(torch.argmax(temp_logits, dim=1), target_label)
    loss= F.cross_entropy(temp_logits, target_label.long()).to(cuda)
    return loss

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
    
def embedding_dist(x1, x2):
    if args.pos_metric == 'l1':
        return l1_dist(x1, x2)
    elif args.pos_metric == 'l2':        
        return l2_dist(x1, x2)
    elif args.pos_metric == 'cos':
        return cosine_similarity( x1, x2 )
    

# def wasserstein_penalty(  ):

def compute_penalty( model, feature, target_label, domain_label):
    curr_domains= np.unique(domain_label)
    ret= torch.tensor(0.).to(cuda)
    for domain in curr_domains:
        indices= domain_label == domain
        temp_logits= model(feature[indices])
        labels= target_label[indices]
        scale = torch.tensor(1.).to(cuda).requires_grad_()
        loss = F.cross_entropy(temp_logits*scale, labels.long()).to(cuda)
        g = grad(loss, [scale], create_graph=True)[0].to(cuda)
        # Since g is scalar output, do we need torch.sum?
        ret+= torch.sum(g**2)
        
    return ret 

def init_data_match_dict(keys, vals, variation):
    data={}
    for key in keys:
        data[key]={}
        if variation:
            val_dim= vals[key]
        else:
            val_dim= vals
            
        if args.dataset == 'color_mnist':        
            data[key]['data']=torch.rand((val_dim, 2, 28, 28))
        elif args.dataset == 'rot_mnist' or args.dataset == 'fashion_mnist':   
            data[key]['data']=torch.rand((val_dim, 1, 224, 224))      
        elif args.dataset == 'pacs':
            data[key]['data']=torch.rand((val_dim, 3, 227, 227))
        
        data[key]['label']=torch.rand((val_dim, 1))
        data[key]['idx']=torch.randint(0, 1, (val_dim, 1))
    return data

def perfect_match_score(indices_matched):
    counter=0
    score=0
    for key in indices_matched:
        for match in indices_matched[key]:
            if key == match:
                score+=1
            counter+=1
    if counter:
        return 100*score/counter
    else:
        return 0

def get_dataloader(train_data_obj, val_data_obj, test_data_obj):
    # Load supervised training
    train_dataset = data_utils.DataLoader(train_data_obj, batch_size=args.batch_size, shuffle=True, **kwargs )
    
    # Can select a higher batch size for val and test domains
    test_batch=512
    val_dataset = data_utils.DataLoader(val_data_obj, batch_size=test_batch, shuffle=True, **kwargs )
    test_dataset = data_utils.DataLoader(test_data_obj, batch_size=test_batch, shuffle=True, **kwargs )

#     elif args.dataset == 'color_mnist' or args.dataset =='rot_color_mnist':
#         # Load supervised training
#         train_dataset = data_utils.DataLoader(                                                                                                   MnistRotated(train_domains, -1, 0.25, 1, 'data/rot_mnist', train=True), batch_size=args.batch_size, shuffle=True, **kwargs )

#         test_dataset = data_utils.DataLoader(                                                                                                   MnistRotated(test_domains, -1, 0.25, 1, 'data/rot_mnist', train=True), batch_size=args.batch_size, shuffle=True, **kwargs )    
    
    return train_dataset, val_dataset, test_dataset


def get_matched_pairs (train_dataset, domain_size, total_domains, training_list_size, phi, match_case, inferred_match):        
    
    #Making Data Matched pairs
    data_matched= init_data_match_dict( range(domain_size), total_domains, 0 )
    domain_data= init_data_match_dict( range(total_domains), training_list_size, 1)
    indices_matched={}
    for key in range(domain_size):
        indices_matched[key]=[]
    perfect_match_rank=[]
        
    domain_count={}
    for domain in range(total_domains):
        domain_count[domain]= 0
        
    # Create dictionary: class label -> list of ordered indices
    for batch_idx, (x_e, y_e ,d_e, idx_e) in enumerate(train_dataset):
        x_e= x_e
        y_e= torch.argmax(y_e, dim=1)
        d_e= torch.argmax(d_e, dim=1).numpy()
        
        domain_indices= np.unique(d_e)
        for domain_idx in domain_indices:                        
            indices= d_e == domain_idx
            ordered_indices= idx_e[indices]
            for idx in range(ordered_indices.shape[0]):                
                #Matching points across domains
                perfect_indice= ordered_indices[idx].item()
                domain_data[domain_idx]['data'][perfect_indice]= x_e[indices][idx] 
                domain_data[domain_idx]['label'][perfect_indice]= y_e[indices][idx]
                domain_data[domain_idx]['idx'][perfect_indice]= idx_e[indices][idx]
                domain_count[domain_idx]+= 1        
    
    #Sanity Check: To check if the domain_data was updated for all the data points
#     for domain in range(total_domains):
#         if domain_count[domain] != training_list_size[domain]:
#             print('Error: Some data points are missing from domain_data dictionary')
    
    # Creating the random permutation tensor for each domain
    perm_size= int(domain_size*(1-match_case))
    
    #Determine the base_domain_idx as the domain with the max samples of the current class
    base_domain_dict={}
    for y_c in range(args.out_classes):
        base_domain_size=0
        base_domain_idx=-1
        for domain_idx in range(total_domains):
            class_idx= domain_data[domain_idx]['label'] == y_c
            curr_size= domain_data[domain_idx]['label'][class_idx].shape[0]
            if base_domain_size < curr_size:
                base_domain_size= curr_size
                base_domain_idx= domain_idx 
                    
        base_domain_dict[y_c]= base_domain_idx
        #print('Base Domain: ', base_domain_size, base_domain_idx, y_c )    
    
    # Applying the random permutation tensor
    for domain_idx in range(total_domains):                        
        total_rand_counter=0
        total_data_idx=0
        for y_c in range(args.out_classes):
            base_domain_idx= base_domain_dict[y_c]            
 
            indices_base= domain_data[base_domain_idx]['label'] == y_c
            indices_base= indices_base[:,0]
            ordered_base_indices= domain_data[base_domain_idx]['idx'][indices_base]        
            
            indices_curr= domain_data[domain_idx]['label'] == y_c
            indices_curr= indices_curr[:,0]                        
            ordered_curr_indices= domain_data[domain_idx]['idx'][indices_curr]
            curr_size= ordered_curr_indices.shape[0]
            
            # Sanity check for perfect match case:
#             if args.perfect_match:
#                 if not torch.equal(ordered_base_indices, ordered_curr_indices):
#                     print('Error: Different indices across domains for perfect match' )
            
            # Only for the perfect match case to generate x% correct match strategy
            rand_base_indices= ordered_base_indices[ ordered_base_indices < perm_size ]
            idx_perm= torch.randperm( rand_base_indices.shape[0] )
            rand_base_indices= rand_base_indices[idx_perm]
            rand_counter=0
                            
            base_feat_data=domain_data[base_domain_idx]['data'][indices_base]
            base_feat_data_split= torch.split( base_feat_data, args.batch_size, dim=0 )
            base_feat=[]
            for batch_feat in base_feat_data_split:
                with torch.no_grad():
                    batch_feat=batch_feat.to(cuda)
                    out= phi(batch_feat)
                    base_feat.append(out.cpu())
            base_feat= torch.cat(base_feat)
            
            if inferred_match:
                feat_x_data= domain_data[domain_idx]['data'][indices_curr]
                feat_x_data_split= torch.split(feat_x_data, args.batch_size, dim=0)
                feat_x=[]
                for batch_feat in feat_x_data_split:
                    with torch.no_grad():
                        batch_feat= batch_feat.to(cuda)
                        out= phi(batch_feat)
                        feat_x.append(out.cpu())
                feat_x= torch.cat(feat_x)                
            
            base_feat= base_feat.unsqueeze(1)    
            base_feat_split= torch.split(base_feat, args.batch_size, dim=0)   
            
            data_idx=0
            for batch_feat in base_feat_split:
                
                if inferred_match:
                    # Need to compute over batches of base_fear due ot CUDA Memory out errors
                    # Else no ned for loop over base_feat_split; could have simply computed feat_x - base_feat
                    ws_dist= torch.sum( (feat_x - batch_feat)**2, dim=2)
                    match_idx= torch.argmin( ws_dist, dim=1 )
                    sort_val, sort_idx= torch.sort( ws_dist, dim=1 )                                    
                    del ws_dist
                                    
                for idx in range(batch_feat.shape[0]):
                    perfect_indice= ordered_base_indices[data_idx].item()
                    
                    if domain_idx == base_domain_idx:
                        curr_indice=  perfect_indice 
                    else:
                        if args.perfect_match:                        
                            if inferred_match:
                                curr_indice= ordered_curr_indices[match_idx[idx]].item()
                                #Find where does the perfect match lies in the sorted order of matches
                                #In the situations where the perfect match is known; the ordered_curr_indices and ordered_base_indices are the same
                                perfect_match_rank.append( (ordered_curr_indices[sort_idx[idx]] == perfect_indice).nonzero()[0,0].item() )
                            else:
                                # To allow x% match case type permutations for datasets where the perfect match is known
                                # In perfect match settings; same ordered indice implies perfect match across domains
                                if perfect_indice < perm_size:
                                    curr_indice= rand_base_indices[rand_counter].item()
                                    rand_counter+=1
                                    total_rand_counter+=1
                                else:
                                    curr_indice=  perfect_indice                      
                            indices_matched[perfect_indice].append(curr_indice)                      
                            
                        else:
                            if inferred_match:
                                curr_indice= ordered_curr_indices[match_idx[idx]].item()                                
                            else: 
                                curr_indice= ordered_curr_indices[data_idx%curr_size].item()

                    data_matched[total_data_idx]['data'][domain_idx]= domain_data[domain_idx]['data'][curr_indice]
                    data_matched[total_data_idx]['label'][domain_idx]= domain_data[domain_idx]['label'][curr_indice]
                    data_idx+=1
                    total_data_idx+=1
        
#         if total_data_idx != domain_size:
#             print('Error: Some data points left from data_matched dictionary', total_data_idx, domain_size)
            
#         if args.perfect_match and inferred_match ==0 and domain_idx != base_domain_idx and  total_rand_counter < perm_size:
#             print('Error: Total random changes made are less than perm_size for domain', domain_idx, total_rand_counter, perm_size)
                            
                
    # Sanity Check:  N keys; K vals per key
#     for key in data_matched.keys():
#         if data_matched[key]['label'].shape[0] != total_domains:
#             print('Error with data matching')

    #Sanity Check: Ensure paired points have the same class label
    wrong_case=0
    for key in data_matched.keys():
        for d_i in range(data_matched[key]['label'].shape[0]):
            for d_j in range(data_matched[key]['label'].shape[0]):
                if d_j > d_i:
                    if data_matched[key]['label'][d_i] != data_matched[key]['label'][d_j]:
                        wrong_case+=1
    #print('Total Label MisMatch across pairs: ', wrong_case )
            
    data_match_tensor=[]
    label_match_tensor=[]
    for key in data_matched.keys():
        data_match_tensor.append( data_matched[key]['data'] )
        label_match_tensor.append(data_matched[key]['label']  )

    data_match_tensor= torch.stack( data_match_tensor ) 
    label_match_tensor= torch.stack( label_match_tensor ) 

    # Creating tensor of shape ( domain_size * total domains, feat size )
    # data_match_tensor= data_match_tensor.view( data_match_tensor.shape[0]*data_match_tensor.shape[1], data_match_tensor.shape[2], data_match_tensor.shape[3], data_match_tensor.shape[4] )
    # label_match_tensor= label_match_tensor.view( label_match_tensor.shape[0]*label_match_tensor.shape[1] )
    
    #print(data_match_tensor.shape, label_match_tensor.shape)
    
    del domain_data
    del data_matched

    return data_match_tensor, label_match_tensor, indices_matched, perfect_match_rank
    
        
    
    
def test(test_dataset, phi, epoch):
    #Test Env Code
    test_acc= 0.0
    test_size=0
    
    for batch_idx, (x_e, y_e ,d_e, idx_e) in enumerate(test_dataset):
        with torch.no_grad():
            x_e= x_e.to(cuda)
            y_e= torch.argmax(y_e, dim=1).to(cuda)
            d_e = torch.argmax(d_e, dim=1).numpy()       
            #print(type(x_e), x_e.shape, y_e.shape, d_e.shape)        

            #Forward Pass
            out= phi(x_e)
            loss_e= torch.mean(erm_loss(out, y_e))        

            test_acc+= torch.sum( torch.argmax(out, dim=1) == y_e ).item()
            test_size+= y_e.shape[0]
            #print('Test Loss Env : ',  loss_e)
    
    #print('Test Accuracy: Epoch ', epoch, 100*test_acc/test_size ) 
        
    return (100*test_acc/test_size)

def compute_t_sne(dataset, phi):
    t_sne_label={}
    for y_c in range(args.out_classes):
        t_sne_label[y_c]=[]
    
    feat_all=[]
    label_all=[]
    domain_all=[]
    for batch_idx, (x_e, y_e, d_e, idx_e) in enumerate(dataset):
        x_e= x_e.to(cuda)
        y_e= torch.argmax(y_e, dim=1)
        d_e= torch.argmax(d_e, dim=1)
        
        with torch.no_grad():
            feat_all.append( phi(x_e).cpu() )
            label_all.append( y_e )
            domain_all.append( d_e )
    
    feat_all= torch.cat(feat_all)
    label_all= torch.cat(label_all).numpy()
    domain_all= torch.cat(domain_all).numpy()
    
    #t-SNE plots     
    if args.rep_dim > 2:
        t_sne_out= t_sne_plot( feat_all ).tolist() 
    elif args.rep_dim ==2:
        t_sne_out = feat_all.detach().numpy().tolist() 
    else:
        print('Error: Represenation Dimension cannot be less than 2')
        
    #print('T-SNE', np.array(t_sne_out).shape, feat_all.shape, label_all.shape, domain_all.shape)
        
    for idx in range(feat_all.shape[0]):
        key= label_all[idx]
        t_sne_label[key].append( t_sne_out[idx] )
        
    return t_sne_label
    
# Input Parsing
parser = argparse.ArgumentParser(description='PACS')
parser.add_argument('--dataset', type=str, default='rot_mnist')
parser.add_argument('--method_name', type=str, default='erm')
parser.add_argument('--pos_metric', type=str, default='l2')
parser.add_argument('--model_name', type=str, default='alexnet')
parser.add_argument('--opt', type=str, default='sgd')
parser.add_argument('--out_classes', type=int, default=10)
parser.add_argument('--rep_dim', type=int, default=250)
parser.add_argument('--test_domain', type=int, default=0, help='0: In angles; 1: out angles')
parser.add_argument('--perfect_match', type=int, default=0, help='0: No perfect match known (PACS); 1: perfect match known (MNIST)')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epochs', type=int, default=25)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--penalty_w', type=float, default=100)
parser.add_argument('--penalty_s', type=int, default=5)
parser.add_argument('--penalty_ws', type=float, default=0.001)
parser.add_argument('--penalty_same_ctr',type=float, default=0.001)
parser.add_argument('--penalty_diff_ctr',type=float, default=0.001)
parser.add_argument('--penalty_erm',type=float, default=1)
parser.add_argument('--same_margin', type=float, default=1.0)
parser.add_argument('--diff_margin', type=float, default=1.0)
parser.add_argument('--epochs_erm', type=int, default=25)
parser.add_argument('--n_runs_erm', type=int, default=2)
parser.add_argument('--penalty_ws_erm', type=float, default=0.1)
parser.add_argument('--match_case_erm', type=float, default=1.0)
parser.add_argument('--pre_trained',type=int, default=1)
parser.add_argument('--match_flag', type=int, default=1, help='0: No Update to Match Strategy; 1: Updates to Match Strategy')
parser.add_argument('--match_case', type=float, default=1, help='0: Random Match; 1: Perfect Match')
parser.add_argument('--match_interrupt', type=int, default=10)
parser.add_argument('--base_domain_idx', type=int, default=1)
parser.add_argument('--ctr_abl', type=int, default=0, help='0: Randomization til class level ; 1: Randomization completely')
parser.add_argument('--match_abl', type=int, default=0, help='0: Randomization til class level ; 1: Randomization completely')
parser.add_argument('--domain_abl', type=int, default=0, help='0: No Abl; x: Train with x domains only')
parser.add_argument('--erm_base', type=int, default=1, help='0: ERM loss added gradually; 1: ERM weight constant')
parser.add_argument('--ctr_phase', type=int, default=1, help='0: No Metric Learning; 1: Metric Learning')
parser.add_argument('--erm_phase', type=int, default=1, help='0: No ERM Learning; 1: ERM Learning')
parser.add_argument('--domain_gen', type=int, default=1)
parser.add_argument('--save_logs', type=int, default=0)
parser.add_argument('--n_runs', type=int, default=3)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--cuda_device', type=int, default=0 )
parser.add_argument('--test_metric', type=str, default='acc')
args = parser.parse_args()


if args.dataset == 'color_mnist':
    from models.color_mnist import *
    from models.ResNet import *
    from data.color_mnist.mnist_loader import MnistRotated
if args.dataset == 'rot_color_mnist':
    from models.rot_mnist import *
    from data.rot_color_mnist.mnist_loader import MnistRotated    
# elif args.dataset == 'fashion_mnist':
#     from models.rot_mnist import *
#     from data.rot_fashion_mnist.fashion_mnist_loader import MnistRotated
elif args.dataset == 'rot_mnist' or args.dataset == 'fashion_mnist':
#     from models.rot_mnist import *
#     from models.metric_rot_mnist import *
    from models.ResNet import *
    from data.rot_mnist.mnist_loader import MnistRotated
elif args.dataset == 'pacs':
    from models.AlexNet import *
    from models.ResNet import *
    from data.pacs.pacs_loader import PACS

#GPU
cuda= torch.device("cuda:" + str(args.cuda_device))

# Environments
base_data_dir='data/' + args.dataset +'/'
base_logs_dir="results/" + args.dataset +'/'  
base_res_dir="results/" + args.dataset + '/'

if cuda:
    kwargs = {'num_workers': 1, 'pin_memory': False} 
else:
    kwargs= {}

if args.dataset == 'rot_mnist' or args.dataset == 'fashion_mnist':    
    #Train and Test Domains
    if args.test_domain==0:
        test_domains= ["30", "45"]
    elif args.test_domain==1:
        test_domains= ["0", "90"]
    elif args.test_domain==2:
        test_domains=["45"]
    elif args.test_domain==3:
        test_domains=["0"]
    
    if args.domain_abl == 0:
        train_domains= ["0", "15", "30", "45", "60", "75", "90"]
    elif args.domain_abl == 2:
        train_domains= ["30", "45"]
    elif args.domain_abl == 3:
        train_domains= ["30", "45", "60"]
#     train_domains= ["0", "30", "60", "90"]
    for angle in test_domains:
        if angle in train_domains:
            train_domains.remove(angle)        
        
elif args.dataset == 'color_mnist' or args.dataset == 'rot_color_mnist':
    train_domains= [0.1, 0.2]
    test_domains= [0.9]
    
elif args.dataset == 'pacs':
    #Train and Test Domains
    if args.test_domain==0:
        test_domains= ["photo"]
    elif args.test_domain==1:
        test_domains=["art_painting"]
    elif args.test_domain==2:
        test_domains=["cartoon"]
    elif args.test_domain==3:
        test_domains=["sketch"]
    elif args.test_domain==-1:
        test_domains=["sketch"]        
    
    train_domains= ["photo", "art_painting", "cartoon", "sketch"]
    for angle in test_domains:
        if angle in train_domains:
            train_domains.remove(angle)      
            
    if args.test_domain==-1:
        train_domains=test_domains            


    
# Parameters
feature_dim= 28*28
rep_dim= args.rep_dim
num_classes= args.out_classes
pre_trained= args.pre_trained
    
if args.dataset in ['rot_mnist', 'color_mnist', 'fashion_mnist']:
    feature_dim= 28*28
    num_ch=1
    pre_trained=0
#         phi= RepNet( feature_dim, rep_dim)
    if args.erm_base:                     
        phi= get_resnet('resnet18', num_classes, args.erm_base, num_ch, pre_trained).to(cuda)
    elif args.erm_phase:
        phi= get_resnet('resnet18', num_classes, 1, num_ch, pre_trained).to(cuda)        
    else:
        rep_dim=512
        phi= get_resnet('resnet18', rep_dim, args.erm_base, num_ch, pre_trained).to(cuda)
elif args.dataset in ['pacs', 'vlcs']:
    if args.model_name == 'alexnet':        
        if args.erm_base:
            phi= alexnet(num_classes, pre_trained, args.erm_base ).to(cuda)
        elif args.erm_phase:
            phi= alexnet(num_classes, pre_trained, 1 ).to(cuda)            
        else:
            rep_dim= 4096                 
            phi= alexnet(rep_dim, pre_trained, args.erm_base).to(cuda)
    elif args.model_name == 'resnet18':
        num_ch=3
        if args.erm_base:
            phi= get_resnet('resnet18', num_classes, args.erm_base, num_ch, pre_trained).to(cuda)
        elif args.erm_phase:
            phi= get_resnet('resnet18', num_classes, 1, num_ch, pre_trained).to(cuda)            
        else:
            rep_dim= 512                
            phi= get_resnet('resnet18', rep_dim, args.erm_base, num_ch, pre_trained).to(cuda)
#print('Model Archtecture: ', args.model_name)
    

#Main Code
epochs=args.epochs
batch_size=args.batch_size
learning_rate= args.lr
lmd=args.penalty_w
anneal_iter= args.penalty_s


match_flag=args.match_flag
match_interrupt=args.match_interrupt
base_domain_idx= args.base_domain_idx
match_counter=0

match_acc=[]
match_rank=[]
match_top_k=[]
final_acc=[]
val_acc=[]

for run in range(args.n_runs):
    
    #Seed for repoduability
    torch.manual_seed(run*10)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(run*10)    
    
    # Path to save results        
    post_string= str(args.penalty_erm) + '_' +  str(args.penalty_ws) + '_' + str(args.penalty_same_ctr) + '_' + str(args.penalty_diff_ctr) + '_' + str(args.rep_dim) + '_' + str(args.match_case) + '_' + str(args.match_interrupt) + '_' + str(args.match_flag) + '_' + str(args.test_domain) + '_' + str(run) + '_' + args.pos_metric + '_' + args.model_name
    
    for temp_idx in range(1):
        # DataLoader
        if args.dataset in ['pacs', 'vlcs']:
            train_data_obj= PACS(train_domains, 'pacs/train_val_splits/', data_case='train')            
    #         val_data_obj= PACS(train_domains, '/pacs/train_val_splits/', data_case='val')        
            test_data_obj= PACS(test_domains, '/pacs/train_val_splits/', data_case='test')
        elif args.dataset in ['rot_mnist', 'fashion_mnist']:
            train_data_obj=  MnistRotated(args.dataset, train_domains, run, 'data/rot_mnist', data_case='train')
    #         val_data_obj=  MnistRotated(args.dataset, train_domains, run, 'data/rot_mnist', data_case='val')       
            test_data_obj=  MnistRotated(args.dataset, test_domains, run, 'data/rot_mnist', data_case='test')

        test_batch= 512
        train_dataset = data_utils.DataLoader(train_data_obj, batch_size=test_batch, shuffle=True, **kwargs )
        test_dataset = data_utils.DataLoader(test_data_obj, batch_size=test_batch, shuffle=True, **kwargs )

        if args.dataset == 'rot_mnist':
            total_domains= len(train_domains)
            domain_size= 2000       
            training_list_size= []
            base_domain_idx= -1
            for idx in range(total_domains):
                training_list_size.append(domain_size)
        elif args.dataset == 'fashion_mnist':
            total_domains= len(train_domains)
            domain_size= 10000       
            base_domain_idx= -1
            training_list_size= []
            for idx in range(total_domains):
                training_list_size.append(domain_size)

        # Load RepNet from save weights
        if args.erm_base:
            if args.domain_abl==0:
                sub_dir= '/ERM_Base'
            elif args.domain_abl ==2:
                sub_dir= '/ERM_Base/' + train_domains[0] + '_' + train_domains[1]
            elif args.domain_abl ==3:
                sub_dir= '/ERM_Base/' + train_domains[0] + '_' + train_domains[1] + '_' + train_domains[2]
        elif args.ctr_phase:
            if args.domain_abl==0:
                sub_dir= '/CTR'
            elif args.domain_abl ==2:
                sub_dir= '/CTR/' + train_domains[0] + '_' + train_domains[1]
            elif args.domain_abl ==3:
                sub_dir= '/CTR/' + train_domains[0] + '_' + train_domains[1] + '_' + train_domains[2]
        elif args.erm_phase:
            if args.domain_abl==0:
                sub_dir= '/ERM'
            elif args.domain_abl ==2:
                sub_dir= '/ERM/' + train_domains[0] + '_' + train_domains[1]
            elif args.domain_abl ==3:
                sub_dir= '/ERM/' + train_domains[0] + '_' + train_domains[1] + '_' + train_domains[2]

        
        if args.erm_phase:
            
            for run_erm in range(args.n_runs_erm):
            
                save_path= base_res_dir + args.method_name + sub_dir + '/Model_' + post_string + '_' + str(args.penalty_ws_erm) + '_' + str(args.match_case_erm) + '_' + str(run_erm) +  '.pth'   
                phi.load_state_dict( torch.load(save_path, map_location='cpu') )
                phi.eval()
                phi=phi.to(cuda)            
            
                #Testing Phase
                if args.test_metric == 'acc':
                    epoch=0
                    test_acc= test( test_dataset, phi, epoch )
                    final_acc.append( test_acc )        
                
                elif args.test_metric == 'other':

                    inferred_match=1
                    data_match_tensor, label_match_tensor, indices_matched, perfect_match_rank= get_matched_pairs( train_dataset, domain_size, total_domains, training_list_size, phi, args.match_case_erm, inferred_match )                

                    if args.perfect_match:
                        score= perfect_match_score(indices_matched)
                        perfect_match_rank= np.array(perfect_match_rank)            

                        match_acc.append(score)
                        match_rank.append( np.mean(perfect_match_rank) )
                        match_top_k.append( 100*np.sum( perfect_match_rank < 10 )/perfect_match_rank.shape[0] )
            
            
        else:
            
            save_path= base_res_dir + args.method_name + sub_dir + '/Model_' + post_string + '.pth'   
            phi.load_state_dict( torch.load(save_path, map_location='cpu') )
            phi.eval()
            phi=phi.to(cuda)
            
            #Testing Phase
            if args.test_metric == 'acc':
                epoch=0
                test_acc= test( test_dataset, phi, epoch )
                final_acc.append( test_acc )        
            
            elif args.test_metric == 'other':
                inferred_match=1
                data_match_tensor, label_match_tensor, indices_matched, perfect_match_rank= get_matched_pairs( train_dataset, domain_size, total_domains, training_list_size, phi, args.match_case_erm, inferred_match )                

                if args.perfect_match:
                    score= perfect_match_score(indices_matched)
                    perfect_match_rank= np.array(perfect_match_rank)            

                    match_acc.append(score)
                    match_rank.append( np.mean(perfect_match_rank) )
                    match_top_k.append( 100*np.sum( perfect_match_rank < 10 )/perfect_match_rank.shape[0] )

final_acc= np.array(final_acc)
match_acc= np.array(match_acc)
match_rank= np.array(match_rank)
match_top_k= np.array(match_top_k)

print('\n')
print('Done for Model..')

if args.test_metric == 'acc':
    print('Test Accuracy', np.mean(final_acc), np.std(final_acc) )
    
elif args.test_metric == 'other':
    print('Perfect Match Score: ',  np.mean(match_acc), np.std(match_acc)  )                    
    print('Mean Perfect Match Score: ',  np.mean(match_rank), np.std(match_rank) )            
    print('TopK Perfect Match Score: ',  np.mean(match_top_k), np.std(match_top_k)  )            
    
print('\n')
