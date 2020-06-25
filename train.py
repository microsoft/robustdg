import sys
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
parser.add_argument('--tau', type=float, default=0.05)
parser.add_argument('--retain', type=float, default=0, help='0: Train from scratch in ERM Phase; 1: Finetune from CTR Phase in ERM Phase')
parser.add_argument('--cuda_device', type=int, default=0 )
args = parser.parse_args()

if args.dataset == 'rot_mnist' or args.dataset == 'fashion_mnist':
    if args.model_name == 'lenet':
        from data.rot_mnist.mnist_loader_lenet import MnistRotated
    else:
        from data.rot_mnist.mnist_loader import MnistRotated
elif args.dataset == 'pacs':
    from data.pacs.pacs_loader import PACS
    
#GPU
cuda= torch.device("cuda:" + str(args.cuda_device))
if cuda:
    kwargs = {'num_workers': 1, 'pin_memory': False} 
else:
    kwargs= {}

if args.dataset == 'rot_mnist' or args.dataset == 'fashion_mnist':  
    
    if args.model_name == 'lenet':
        #Train and Test Domains
        if args.test_domain==0:
            test_domains= ["0"]
        elif args.test_domain==1:
            test_domains= ["15"]
        elif args.test_domain==2:
            test_domains=["30"]
        elif args.test_domain==3:
            test_domains=["45"]
        elif args.test_domain==4:
            test_domains=["60"]
        elif args.test_domain==5:
            test_domains=["75"]

        if args.domain_abl == 0:
            train_domains= ["0", "15", "30", "45", "60", "75"]
        elif args.domain_abl == 2:
            train_domains= ["30", "45"]
        elif args.domain_abl == 3:
            train_domains= ["30", "45", "60"]
            
        for angle in test_domains:
            if angle in train_domains:
                train_domains.remove(angle)        
        
    else:    
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

final_report_accuracy=[]
base_res_dir="results/" + args.dataset + '/'

for run in range(args.n_runs):
    
    #Seed for repoduability
    torch.manual_seed(run*10)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(run*10)    

    # Path to save results        
    post_string= str(args.penalty_erm) + '_' +  str(args.penalty_ws) + '_' + str(args.penalty_same_ctr) + '_' + str(args.penalty_diff_ctr) + '_' + str(args.rep_dim) + '_' + str(args.match_case) + '_' + str(args.match_interrupt) + '_' + str(args.match_flag) + '_' + str(args.test_domain) + '_' + str(run) + '_' + args.pos_metric + '_' + args.model_name

    
    # DataLoader        
    train_dataset, val_dataset, test_dataset= get_dataloader( train_domains, test_domains )
    total_domains= len(train_domains)
    domain_size= train_data_obj.base_domain_size       
    base_domain_idx= train_data_obj.base_domain_idx
    training_list_size= train_data_obj.training_list_size
    print('Train Domains, Domain Size, BaseDomainIdx, Total Domains: ', train_domains, domain_size, base_domain_idx, total_domains, training_list_size)
        
            if bool_erm:
                #Validation Phase
                test_acc= test( val_dataset, phi, epoch, 'Val' )
                val_acc.append( test_acc )
                #Testing Phase
                test_acc= test( test_dataset, phi, epoch, 'Test' )
                final_acc.append( test_acc )        
                
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
                    
        # Store the weights of the model
        torch.save(phi.state_dict(), base_res_dir + args.method_name +  sub_dir + '/Model_' + post_string + '.pth')        
        
        # Final Report Accuacy
        if args.erm_base:
            final_report_accuracy.append( final_acc[-1] )

    
    if args.erm_phase:
        
        for run_erm in range(args.n_runs_erm):

            # Load RepNet from save weights
            sub_dir='/CTR'
            save_path= base_res_dir + args.method_name + sub_dir + '/Model_' + post_string + '.pth'   
            phi.load_state_dict( torch.load(save_path) )
            phi.eval()
            

            #Inferred Match Case
            if args.match_case_erm == -1:
                inferred_match=1
                data_match_tensor, label_match_tensor, indices_matched, perfect_match_rank= get_matched_pairs( args, train_dataset, domain_size, total_domains, training_list_size, phi, args.match_case_erm, inferred_match )                

            else:
                inferred_match=0
                # x% percentage match initial strategy
                data_match_tensor, label_match_tensor, indices_matched, perfect_match_rank= get_matched_pairs( args, train_dataset, domain_size, total_domains, training_list_size, phi, args.match_case_erm, inferred_match )                
                    
            # Model and parameters
            if args.retain:
                phi_erm= ClfNet( phi, rep_dim, num_classes ).to(cuda)
            else:
                if args.dataset in ['rot_mnist', 'color_mnist', 'fashion_mnist']:
                    feature_dim= 28*28
                    num_ch=1
                    pre_trained=0
                    if args.model_name == 'lenet':
                        phi_erm= LeNet5().to(cuda)
                    else:
                        phi_erm= get_resnet('resnet18', num_classes, 1, num_ch, pre_trained).to(cuda)
                    
                elif args.dataset in ['pacs', 'vlcs']:
                    if args.model_name == 'alexnet':        
                        phi_erm= alexnet(num_classes, pre_trained, 1 ).to(cuda)
                    elif args.model_name == 'resnet18':
                        num_ch=3
                        phi_erm= get_resnet('resnet18', num_classes, 1, num_ch, pre_trained).to(cuda)                

            learning_rate=args.lr
            opt= optim.SGD([
                {'params': filter(lambda p: p.requires_grad, phi_erm.parameters()) }                
            ], lr=learning_rate, weight_decay=5e-4, momentum=0.9)
                        
                    
            #Training and Evaludation
            final_acc=[]
            val_acc=[]
            for epoch in range(args.epochs_erm):

                #Train Specifications
                bool_erm=1
                bool_ws=1
                bool_ctr=0            
                #Train
                penalty_erm, penalty_irm, penalty_ws, penalty_same_ctr, penalty_diff_ctr = train( train_dataset, data_match_tensor, label_match_tensor, phi_erm, opt, opt_ws, scheduler, epoch, base_domain_idx, bool_erm, bool_ws, bool_ctr ) 
                #Test
                #Validation Phase
                test_acc= test( val_dataset, phi_erm, epoch, 'Val' )
                val_acc.append( test_acc )
                #Testing Phase
                test_acc= test( test_dataset, phi_erm, epoch, 'Test' )
                final_acc.append(test_acc)
            
            post_string_erm= str(args.penalty_erm) + '_' +  str(args.penalty_ws) + '_' + str(args.penalty_same_ctr) + '_' + str(args.penalty_diff_ctr) + '_' + str(args.rep_dim)  + '_' + str(args.match_case) + '_' + str(args.match_interrupt) + '_' + str(args.match_flag) + '_' + str(args.test_domain) + '_' + str(run) + '_' + args.pos_metric + '_' + args.model_name + '_' + str(args.penalty_ws_erm) + '_' + str(args.match_case_erm) + '_' + str(run_erm)
            
            final_acc= np.array(final_acc)

            if args.domain_abl==0:
                sub_dir= '/ERM'
            elif args.domain_abl ==2:
                sub_dir= '/ERM/' + train_domains[0] + '_' + train_domains[1]
            elif args.domain_abl ==3:
                sub_dir= '/ERM/' + train_domains[0] + '_' + train_domains[1] + '_' + train_domains[2]
                        
            np.save( base_res_dir + args.method_name + sub_dir + '/ACC_' + post_string_erm + '.npy' , final_acc )
            
            # Store the weights of the model
            torch.save(phi_erm.state_dict(), base_res_dir + args.method_name +  sub_dir + '/Model_' + post_string_erm + '.pth')
            
            # Final Report Accuracy
            if args.erm_phase:
                final_report_accuracy.append( final_acc[-1] )

if args.erm_base or args.erm_phase:
    print('\n')
    print('Done for the Model..')

    print('Final Test Accuracy', np.mean(final_report_accuracy), np.std(final_report_accuracy) )

    print('\n')