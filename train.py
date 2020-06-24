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
            
def train( train_dataset, data_match_tensor, label_match_tensor, phi, opt, opt_ws, scheduler, epoch, base_domain_idx, bool_erm, bool_ws, bool_ctr):
    
    penalty_erm=0
    penalty_erm_2=0
    penalty_irm=0
    penalty_ws=0
    penalty_same_ctr=0
    penalty_diff_ctr=0
    penalty_same_hinge=0
    penalty_diff_hinge=0
    train_acc= 0.0
    train_size=0
    
    perm = torch.randperm(data_match_tensor.size(0))            
    data_match_tensor_split= torch.split(data_match_tensor[perm], args.batch_size, dim=0)
    label_match_tensor_split= torch.split(label_match_tensor[perm], args.batch_size, dim=0)
    print('Split Matched Data: ', len(data_match_tensor_split), data_match_tensor_split[0].shape, len(label_match_tensor_split))
    
    #Loop Over One Environment
    for batch_idx, (x_e, y_e ,d_e, idx_e) in enumerate(train_dataset):
#         print('Batch Idx: ', batch_idx)
        
        opt.zero_grad()
        opt_ws.zero_grad()
        
        x_e= x_e.to(cuda)
        y_e= torch.argmax(y_e, dim=1).to(cuda)
        d_e= torch.argmax(d_e, dim=1).numpy()

        #Forward Pass
        out= phi(x_e)
        loss_e= torch.tensor(0.0).to(cuda)
        penalty_e= torch.tensor(0.0).to(cuda)
                
        if bool_erm:
            # torch.mean not really required here since the reduction mean is set by default in ERM loss
            if args.method_name in ['erm', 'irm'] or ( args.erm_phase==1 and args.match_case_erm == -1 ):
                loss_e= erm_loss(out, y_e)
            else:
                #####
                ## Experimenting for now to keep standard erm loss for all the different methods
                ####
                loss_e= 0*erm_loss(out, y_e)
            
#             penalty_e= compute_penalty(phi, x_e, y_e, d_e)                    
#             penalty_erm+= float(loss_e)
#             penalty_irm+= float(penalty_e)

#             weight_norm = torch.tensor(0.).to(cuda)
#             for w in phi.erm_net[-1].parameters():
#                 weight_norm += w.norm().pow(2)

            if epoch > anneal_iter:
                loss_e+= lmd*penalty_e
                if lmd > 1.0:
                  # Rescale the entire loss to keep gradients in a reasonable range
                  loss_e /= lmd            

#             loss_e+=0.001*weight_norm        
    
        wasserstein_loss=torch.tensor(0.0).to(cuda)
        erm_loss_2= torch.tensor(0.0).to(cuda) 
        same_ctr_loss = torch.tensor(0.0).to(cuda)
        diff_ctr_loss = torch.tensor(0.0).to(cuda)
        same_hinge_loss = torch.tensor(0.0).to(cuda)
        diff_hinge_loss = torch.tensor(0.0).to(cuda)
        if epoch > anneal_iter and args.method_name in ['rep_match', 'phi_match', 'phi_match_abl']:
#             sample_size= args.batch_size
#             perm = torch.randperm(data_match_tensor.size(0))
#             idx = perm[:sample_size]
            
            
            # To cover the varying size of the last batch for data_match_tensor_split, label_match_tensor_split
            total_batch_size= len(data_match_tensor_split)
            if batch_idx >= total_batch_size:
                break
            curr_batch_size= data_match_tensor_split[batch_idx].shape[0]

#             data_match= data_match_tensor[idx].to(cuda)
            data_match= data_match_tensor_split[batch_idx].to(cuda)
            data_match= data_match.view( data_match.shape[0]*data_match.shape[1], data_match.shape[2], data_match.shape[3], data_match.shape[4] )            
            feat_match= phi( data_match )
            
#             label_match= label_match_tensor[idx].to(cuda)           
            label_match= label_match_tensor_split[batch_idx].to(cuda)
            label_match= label_match.view( label_match.shape[0]*label_match.shape[1] )
        
            if bool_erm:
                erm_loss_2+= erm_loss(feat_match, label_match)
                penalty_erm_2+= float(erm_loss_2)                
            
            if args.method_name=="rep_match":
                temp_out= phi.predict_conv_net( data_match )
                temp_out= temp_out.view(-1, temp_out.shape[1]*temp_out.shape[2]*temp_out.shape[3])
                feat_match= phi.predict_fc_net(temp_out)
                del temp_out
            
            # Creating tensor of shape ( domain size, total domains, feat size )
            if len(feat_match.shape) == 4:
                feat_match= feat_match.view( curr_batch_size, len(train_domains), feat_match.shape[1]*feat_match.shape[2]*feat_match.shape[3] )
            else:
                 feat_match= feat_match.view( curr_batch_size, len(train_domains), feat_match.shape[1] )
                    
            label_match= label_match.view( curr_batch_size, len(train_domains) )

    #             print(feat_match.shape)
            data_match= data_match.view( curr_batch_size, len(train_domains), data_match.shape[1], data_match.shape[2], data_match.shape[3] )    
        
            #Positive Match Loss
            if bool_ws:
                pos_match_counter=0
                for d_i in range(feat_match.shape[1]):
    #                 if d_i != base_domain_idx:
    #                     continue
                    for d_j in range(feat_match.shape[1]):
                        if d_j > d_i:                        
                    
                            if args.erm_phase:
                                wasserstein_loss+= torch.sum( torch.sum( (feat_match[:, d_i, :] - feat_match[:, d_j, :])**2, dim=1 ) ) 
                            else:
                                if args.pos_metric == 'l2':
                                    wasserstein_loss+= torch.sum( torch.sum( (feat_match[:, d_i, :] - feat_match[:, d_j, :])**2, dim=1 ) ) 
                                elif args.pos_metric == 'l1':
                                    wasserstein_loss+= torch.sum( torch.sum( torch.abs(feat_match[:, d_i, :] - feat_match[:, d_j, :]), dim=1 ) )        
                                elif args.pos_metric == 'cos':
                                    wasserstein_loss+= torch.sum( cosine_similarity( feat_match[:, d_i, :], feat_match[:, d_j, :] ) )

                            pos_match_counter += feat_match.shape[0]
                
                wasserstein_loss = wasserstein_loss / pos_match_counter
                penalty_ws+= float(wasserstein_loss)                            
                
            # Contrastive Loss
            if bool_ctr:
                same_neg_counter=1
                diff_neg_counter=1
                for y_c in range(args.out_classes):
                    
                    pos_indices= label_match[:, 0] == y_c
                    neg_indices= label_match[:, 0] != y_c
                    pos_feat_match= feat_match[pos_indices]
                    neg_feat_match= feat_match[neg_indices]
                    
                    if pos_feat_match.shape[0] > neg_feat_match.shape[0]:
                        print('Weird! Positive Matches are more than the negative matches?', pos_feat_match.shape[0], neg_feat_match.shape[0])
                        
                    # If no instances of label y_c in the current batch then continue
                    if pos_feat_match.shape[0] ==0 or neg_feat_match.shape[0] == 0:
                        continue
                        
                    # Iterating over anchors from different domains
                    for d_i in range(pos_feat_match.shape[1]):
                        if torch.sum( torch.isnan(neg_feat_match) ):
                            print('Non Reshaped X2 is Nan')
                            sys.exit()

                        diff_neg_feat_match= neg_feat_match.view(  neg_feat_match.shape[0]*neg_feat_match.shape[1], neg_feat_match.shape[2] )
                        
                        if torch.sum( torch.isnan(diff_neg_feat_match) ):
                            print('Reshaped X2 is Nan')
                            sys.exit()
                            
                        neg_dist= embedding_dist( pos_feat_match[:, d_i, :], diff_neg_feat_match[:, :], args.tau, xent=True)     
                        if torch.sum(torch.isnan(neg_dist)):
                            print('Neg Dist Nan')
                            sys.exit()
                        
                        # Iterating pos dist for current anchor
                        for d_j in range(pos_feat_match.shape[1]):
                            if d_i != d_j:
                                pos_dist= 1.0 - embedding_dist( pos_feat_match[:, d_i, :], pos_feat_match[:, d_j, :] )
                                pos_dist= pos_dist / args.tau
                                if torch.sum(torch.isnan(neg_dist)):
                                    print('Pos Dist Nan')
                                    sys.exit()
                                    
                                if torch.sum( torch.isnan( torch.log( torch.exp(pos_dist) + neg_dist ) ) ):
                                    print('Xent Nan')
                                    sys.exit()

#                                 print( 'Pos Dist', pos_dist )
#                                 print( 'Log Dist ', torch.log( torch.exp(pos_dist) + neg_dist ))
                                diff_hinge_loss+= -1*torch.sum( pos_dist - torch.log( torch.exp(pos_dist) + neg_dist ) )                                 
                                diff_ctr_loss+= torch.sum(neg_dist)
                                diff_neg_counter+= pos_dist.shape[0]
                        
                same_ctr_loss = same_ctr_loss / same_neg_counter
                diff_ctr_loss = diff_ctr_loss / diff_neg_counter
                same_hinge_loss = same_hinge_loss / same_neg_counter
                diff_hinge_loss = diff_hinge_loss / diff_neg_counter      

                penalty_same_ctr+= float(same_ctr_loss)
                penalty_diff_ctr+= float(diff_ctr_loss)
                penalty_same_hinge+= float(same_hinge_loss)
                penalty_diff_hinge+= float(diff_hinge_loss)
            
            if args.erm_base:
                if epoch >= args.match_interrupt and args.match_flag==1:
                    loss_e += ( args.penalty_ws*( epoch - anneal_iter - args.match_interrupt )/(args.epochs - anneal_iter - args.match_interrupt) )*wasserstein_loss
                    loss_e += ( args.penalty_same_ctr*( epoch - anneal_iter - args.match_interrupt )/(args.epochs - anneal_iter - args.match_interrupt) )*same_hinge_loss                    
                    loss_e += ( args.penalty_diff_ctr*( epoch - anneal_iter - args.match_interrupt )/(args.epochs - anneal_iter - args.match_interrupt) )*diff_hinge_loss
                else:
                    loss_e += ( args.penalty_ws*( epoch-anneal_iter )/(args.epochs -anneal_iter) )*wasserstein_loss
                    loss_e += ( args.penalty_same_ctr*( epoch-anneal_iter )/(args.epochs -anneal_iter) )*same_hinge_loss
                    loss_e += ( args.penalty_diff_ctr*( epoch-anneal_iter )/(args.epochs -anneal_iter) )*diff_hinge_loss

                loss_e += args.penalty_erm*erm_loss_2            
            
            # No CTR and No Match Update Case here
            elif args.erm_phase:
                loss_e += ( args.penalty_ws_erm*( epoch-anneal_iter )/(args.epochs_erm -anneal_iter) )*wasserstein_loss
                loss_e += args.penalty_erm*erm_loss_2            
                
            elif args.ctr_phase:
                if epoch >= args.match_interrupt:
                    loss_e += ( args.penalty_ws*( epoch - anneal_iter - args.match_interrupt )/(args.epochs - anneal_iter - args.match_interrupt) )*wasserstein_loss
#                     loss_e += ( args.penalty_same_ctr*( epoch - anneal_iter - args.match_interrupt )/(args.epochs - anneal_iter - args.match_interrupt) )*same_hinge_loss
                loss_e += ( args.penalty_diff_ctr*( epoch-anneal_iter )/(args.epochs -anneal_iter) )*diff_hinge_loss
                        
        loss_e.backward(retain_graph=False)
        opt.step()
        
#         opt.zero_grad()
#         opt_ws.zero_grad()
#         wasserstein_loss.backward()
#         opt_ws.step()
        
        del penalty_e
        del erm_loss_2
        del wasserstein_loss 
        del same_ctr_loss
        del diff_ctr_loss
        del same_hinge_loss
        del diff_hinge_loss
        del loss_e
        torch.cuda.empty_cache()
        
        if bool_erm:
            train_acc+= torch.sum(torch.argmax(out, dim=1) == y_e ).item()
        train_size+= y_e.shape[0]
                
   
    print('Train Loss Basic : ',  penalty_erm, penalty_irm, penalty_ws, penalty_erm_2 )
    print('Train Loss Ctr : ', penalty_same_ctr, penalty_diff_ctr, penalty_same_hinge, penalty_diff_hinge)
    if bool_erm:
        print('Train Acc Env : ', 100*train_acc/train_size )
    print('Done Training for epoch: ', epoch)                
    
#     scheduler.step()
        
    return penalty_erm_2, penalty_irm, penalty_ws, penalty_same_ctr, penalty_diff_ctr
    
    
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
    if args.model_name == 'lenet':
        from models.LeNet import *
        from models.ResNet import *
        from data.rot_mnist.mnist_loader_lenet import MnistRotated
#         from data.rot_mnist.mnist_loader import MnistRotated
    else:
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
for run in range(args.n_runs):
    
    #Seed for repoduability
    torch.manual_seed(run*10)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(run*10)    

    # Path to save results        
    post_string= str(args.penalty_erm) + '_' +  str(args.penalty_ws) + '_' + str(args.penalty_same_ctr) + '_' + str(args.penalty_diff_ctr) + '_' + str(args.rep_dim) + '_' + str(args.match_case) + '_' + str(args.match_interrupt) + '_' + str(args.match_flag) + '_' + str(args.test_domain) + '_' + str(run) + '_' + args.pos_metric + '_' + args.model_name

    
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
            if args.model_name == 'lenet':
                phi= LeNet5().to(cuda)
            else:
                phi= get_resnet('resnet18', num_classes, args.erm_base, num_ch, pre_trained).to(cuda)
        else:
            rep_dim=512
            phi= get_resnet('resnet18', rep_dim, args.erm_base, num_ch, pre_trained).to(cuda)
    elif args.dataset in ['pacs', 'vlcs']:
        if args.model_name == 'alexnet':        
            if args.erm_base:
                phi= alexnet(num_classes, pre_trained, args.erm_base ).to(cuda)
            else:
                rep_dim= 4096                 
                phi= alexnet(rep_dim, pre_trained, args.erm_base).to(cuda)
        elif args.model_name == 'resnet18':
            num_ch=3
            if args.erm_base:
                phi= get_resnet('resnet18', num_classes, args.erm_base, num_ch, pre_trained).to(cuda)
            else:
                rep_dim= 512                
                phi= get_resnet('resnet18', rep_dim, args.erm_base, num_ch, pre_trained).to(cuda)
    print('Model Archtecture: ', args.model_name)
    
    # Ensure that the rep_dim and the architecture matches
    # Like for alexnet, resnet the rep dim would be pre determined to be the second last layer
    phi_erm= ClfNet(phi, rep_dim, num_classes).to(cuda)

    #Main Code
    epochs=args.epochs
    batch_size=args.batch_size
    learning_rate= args.lr
    lmd=args.penalty_w
    anneal_iter= args.penalty_s

    if args.opt == 'sgd':
        opt= optim.SGD([
                     {'params': filter(lambda p: p.requires_grad, phi.parameters()) }, 
            ], lr= learning_rate, weight_decay= 5e-4, momentum= 0.9,  nesterov=True )        
    elif args.opt == 'adam':
        opt= optim.Adam([
                    {'params': filter(lambda p: p.requires_grad, phi.parameters())},
            ], lr= learning_rate)

    patience= 25
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=patience)
    
    if args.model_name=='alexnet':
        opt_ws= optim.SGD([
                {'params': filter(lambda p: p.requires_grad, phi.classifier[-1].parameters()), 'lr': learning_rate, 'weight_decay': 1e-5, 'momentum': 0.9 },
        ] )

        opt_all= optim.SGD([
                {'params': filter(lambda p: p.requires_grad, phi.classifier[-1].parameters()), 'lr': learning_rate, 'weight_decay': 1e-5, 'momentum': 0.9 },
        ] )
    elif args.model_name=='resnet18' or args.method_name=='resnet50':
        opt_ws= optim.SGD([
                 {'params': filter(lambda p: p.requires_grad, phi.fc.parameters()), 'lr': learning_rate, 'weight_decay': 1e-5, 'momentum': 0.9 },   
        ] )

        opt_all= optim.SGD([
                {'params': filter(lambda p: p.requires_grad, phi.fc.parameters()), 'lr': learning_rate, 'weight_decay': 1e-5, 'momentum': 0.9 },
        ] )
    else:
        opt_ws=opt
        opt_all=opt

    # opt_all= optim.SGD([
    #          {'params': filter(lambda p: p.requires_grad, phi.features.parameters()), 'lr': learning_rate/100, 'weight_decay': 5e-4, 'momentum': 0.9 },
    #         {'params': filter(lambda p: p.requires_grad, phi.classifier[-1].parameters()), 'lr': learning_rate, 'weight_decay': 5e-4, 'momentum': 0.9 },
    #         {'params': filter(lambda p: p.requires_grad, phi.classifier[-4].parameters()), 'lr': learning_rate, 'weight_decay': 5e-4, 'momentum': 0.9 },      
    #         {'params': filter(lambda p: p.requires_grad, phi.classifier[-7].parameters()), 'lr': learning_rate, 'weight_decay': 5e-4, 'momentum': 0.9 },
    #         {'params': filter(lambda p: p.requires_grad, phi.classifier[-10].parameters()), 'lr': learning_rate, 'weight_decay': 5e-4, 'momentum': 0.9 },      
    #         {'params': filter(lambda p: p.requires_grad, phi.classifier[-12].parameters()), 'lr': learning_rate/100, 'weight_decay': 5e-4, 'momentum': 0.9 },   
    #         {'params': filter(lambda p: p.requires_grad, phi.classifier[-15].parameters()), 'lr': learning_rate/100, 'weight_decay': 5e-4, 'momentum': 0.9 }   
    # ] )

    # opt= optim.Adam([
    # #         {'params': filter(lambda p: p.requires_grad, phi.features.parameters()) },
    #         {'params': filter(lambda p: p.requires_grad, phi_erm.rep_net.classifier[-1].parameters()) },
    #         {'params': filter(lambda p: p.requires_grad, phi_erm.erm_net.parameters()) }
    # ], lr=learning_rate)


    loss_erm=[]
    loss_irm=[]
    loss_ws=[]
    loss_same_ctr=[]
    loss_diff_ctr=[]
    match_diff=[]
    match_acc=[]
    match_rank=[]
    match_top_k=[]
    final_acc=[]
    val_acc=[]

    match_flag=args.match_flag
    match_interrupt=args.match_interrupt
    base_domain_idx= args.base_domain_idx
    match_counter=0
    
    # DataLoader
    if args.dataset in ['pacs', 'vlcs']:
        ## TODO: Change the dataloader of PACS to incoporate the val indices
        train_data_obj= PACS(train_domains, '/pacs/train_val_splits/', data_case='train')
        val_data_obj= PACS(train_domains, '/pacs/train_val_splits/', data_case='val')        
        test_data_obj= PACS(test_domains, '/pacs/train_val_splits/', data_case='test')
    elif args.dataset in ['rot_mnist', 'fashion_mnist']:
        train_data_obj=  MnistRotated(args.dataset, train_domains, 3+run, 'data/rot_mnist', data_case='train')
        val_data_obj=  MnistRotated(args.dataset, train_domains, 3+run, 'data/rot_mnist', data_case='val')       
        test_data_obj=  MnistRotated(args.dataset, test_domains, 3+run, 'data/rot_mnist', data_case='test')
        
    train_dataset, val_dataset, test_dataset= get_dataloader( train_data_obj, val_data_obj, test_data_obj )

    total_domains= len(train_domains)
    domain_size= train_data_obj.base_domain_size       
    base_domain_idx= train_data_obj.base_domain_idx
    training_list_size= train_data_obj.training_list_size
    print('Train Domains, Domain Size, BaseDomainIdx, Total Domains: ', train_domains, domain_size, base_domain_idx, total_domains, training_list_size)
        
    
    # Either end to end training fashion (erm_base) or contrastive learning rep phase (ctr_phase)
    if args.erm_base or args.ctr_phase:

        for epoch in range(epochs):    

            if epoch % match_interrupt == 0:
                #Start with initially defined batch; else find the local approximate batch
                if epoch > 0:                    
                    inferred_match=1
                    if args.match_flag and match_counter <100:
                        data_match_tensor, label_match_tensor, indices_matched, perfect_match_rank= get_matched_pairs( args, train_dataset, domain_size, total_domains, training_list_size, phi, args.match_case, inferred_match )
                        match_counter+=1
                        #Reset the weights after very match strategy update
        #                 phi= RotMNIST( feature_dim, num_classes ).to(cuda)
        #                 opt= optim.Adam([
        #                         {'params': filter(lambda p: p.requires_grad, phi.predict_conv_net.parameters()) },
        #                         {'params': filter(lambda p: p.requires_grad, phi.predict_fc_net.parameters()) },
        #                         {'params': filter(lambda p: p.requires_grad, phi.predict_final_net.parameters()) }                   
        #                     ], lr=learning_rate)                

                    elif args.match_flag ==0 or match_counter>=1:
                        temp_1, temp_2, indices_matched, perfect_match_rank= get_matched_pairs( args, train_dataset, domain_size, total_domains, training_list_size, phi, args.match_case, inferred_match )                

                    perfect_match_rank= np.array(perfect_match_rank)
                    if args.perfect_match:
                        print('Mean Perfect Match Score: ', np.mean(perfect_match_rank), 100*np.sum(perfect_match_rank < 10)/perfect_match_rank.shape[0] )
                        match_rank.append( np.mean(perfect_match_rank) )
                        match_top_k.append( 100*np.sum( perfect_match_rank < 10 )/perfect_match_rank.shape[0] )

                else:
                    inferred_match=0
                    data_match_tensor, label_match_tensor, indices_matched, perfect_match_rank= get_matched_pairs( args, train_dataset, domain_size, total_domains, training_list_size, phi, args.match_case, inferred_match )

                ## To ensure a random match keeps happening after every match interrupt
    #             data_match_tensor, label_match_tensor, indices_matched= get_matched_pairs( args, train_dataset, domain_size, total_domains, base_domain_idx, args.match_case )          
                if args.perfect_match:
                    score= perfect_match_score(indices_matched)
                    print('Perfect Match Score: ', score)
                    match_acc.append(score)

            # Decide which losses to optimizer depending on the end to end case (erm_base==1) or block wise (erm_base=0)
            if args.erm_base:
                bool_erm=1
                bool_ws=1
                bool_ctr=0
            else:
                bool_erm=0
                bool_ws=1
                bool_ctr=1

            # To decide which till which layer to finetune
            if epoch > -1:
                penalty_erm, penalty_irm, penalty_ws, penalty_same_ctr, penalty_diff_ctr = train( train_dataset, data_match_tensor, label_match_tensor, phi, opt, opt_ws, scheduler, epoch, base_domain_idx, bool_erm, bool_ws, bool_ctr )
            else:
                penalty_erm, penalty_irm, penalty_ws, penalty_same_ctr, penalty_diff_ctr= train( train_dataset, data_match_tensor, label_match_tensor, phi, opt_all, opt_ws, epoch, base_domain_idx, bool_erm, bool_ws, bool_ctr )       

            loss_erm.append( penalty_erm )
            loss_irm.append( penalty_irm )
            loss_ws.append( penalty_ws )
            loss_same_ctr.append( penalty_same_ctr )
            loss_diff_ctr.append( penalty_diff_ctr )

            if bool_erm:
                #Validation Phase
                test_acc= test( val_dataset, phi, epoch, 'Val' )
                val_acc.append( test_acc )
                #Testing Phase
                test_acc= test( test_dataset, phi, epoch, 'Test' )
                final_acc.append( test_acc )        
                
        loss_erm= np.array(loss_erm)
        loss_irm= np.array(loss_irm)
        loss_ws= np.array(loss_ws)
        loss_same_ctr= np.array(loss_same_ctr)
        loss_diff_ctr= np.array(loss_diff_ctr)
        final_acc= np.array(final_acc)
        val_acc= np.array(val_acc)
        match_rank= np.array(match_rank)
        match_top_k= np.array(match_top_k)
    
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
                    
        np.save( base_res_dir + args.method_name + sub_dir + '/ERM_' + post_string + '.npy' , loss_erm )
        np.save( base_res_dir + args.method_name + sub_dir + '/WS_' + post_string + '.npy', loss_ws )
        np.save( base_res_dir + args.method_name + sub_dir + '/S_CTR_' + post_string + '.npy', loss_same_ctr )
        np.save( base_res_dir + args.method_name + sub_dir +'/D_CTR_' + post_string + '.npy', loss_diff_ctr )
        np.save( base_res_dir + args.method_name + sub_dir +'/ACC_' + post_string + '.npy', final_acc )
        np.save( base_res_dir + args.method_name + sub_dir +'/Val_' + post_string + '.npy', val_acc )
        

        if args.perfect_match:
            np.save( base_res_dir + args.method_name +  sub_dir +'/Match_Acc_' + post_string + '.npy', match_acc )
            np.save( base_res_dir + args.method_name +  sub_dir +'/Match_Rank_' + post_string + '.npy', match_rank )
            np.save( base_res_dir + args.method_name +  sub_dir +'/Match_TopK_' + post_string + '.npy', match_top_k )

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

                if args.perfect_match:
                    score= perfect_match_score(indices_matched)
                    print('Perfect Match Score: ', score)                    
                    perfect_match_rank= np.array(perfect_match_rank)            
                    print('Mean Perfect Match Score: ', np.mean(perfect_match_rank), 100*np.sum(perfect_match_rank < 10)/perfect_match_rank.shape[0] )

            else:
                inferred_match=0
                # x% percentage match initial strategy
                data_match_tensor, label_match_tensor, indices_matched, perfect_match_rank= get_matched_pairs( args, train_dataset, domain_size, total_domains, training_list_size, phi, args.match_case_erm, inferred_match )                
               
                if args.perfect_match:
                    score= perfect_match_score(indices_matched)
                    print('Perfect Match Score: ', score)                    
                    perfect_match_rank= np.array(perfect_match_rank)            
                    print('Mean Perfect Match Score: ', np.mean(perfect_match_rank), 100*np.sum(perfect_match_rank < 10)/perfect_match_rank.shape[0] )

                    
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