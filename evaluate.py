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

    
#Validation Phase
test_acc= test( val_dataset, phi, epoch, 'Val' )
val_acc.append( test_acc )
#Testing Phase
test_acc= test( test_dataset, phi, epoch, 'Test' )
final_acc.append( test_acc )        


def test(test_dataset, phi, epoch, case='Test'):
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
    
    print( case + ' Accuracy: Epoch ', epoch, 100*test_acc/test_size ) 
        
    return (100*test_acc/test_size)
    

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
