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

class MatchDG(BaseAlgo):
    def __init__(self, args, train_dataset, train_domains, total_domains, domain_size, training_list_size, ctr_phase=1):
        
        super().__init__() 

    def train():
        # Initialise and call train functions depending on the method's phase
        
        
    def train_ctr_phase(self):
        for epoch in range(epochs):    
            
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
    
            #Batch iteration over single epoch
            for batch_idx, (x_e, y_e ,d_e, idx_e) in enumerate(train_dataset):
        #         print('Batch Idx: ', batch_idx)

                opt.zero_grad()
                loss_e= torch.tensor(0.0).to(cuda)            

                x_e= x_e.to(cuda)
                y_e= torch.argmax(y_e, dim=1).to(cuda)
                d_e= torch.argmax(d_e, dim=1).numpy()

                same_ctr_loss = torch.tensor(0.0).to(cuda)
                diff_ctr_loss = torch.tensor(0.0).to(cuda)
                same_hinge_loss = torch.tensor(0.0).to(cuda)
                diff_hinge_loss = torch.tensor(0.0).to(cuda)
                if epoch > anneal_iter:
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

                    # Contrastive Loss
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
                
                    loss_e += ( args.penalty_diff_ctr*( epoch-anneal_iter )/(args.epochs -anneal_iter) )*diff_hinge_loss
                        
                loss_e.backward(retain_graph=False)
                opt.step()
                
                del same_ctr_loss
                del diff_ctr_loss
                del same_hinge_loss
                del diff_hinge_loss
                torch.cuda.empty_cache()
   
            print('Train Loss Ctr : ', penalty_same_ctr, penalty_diff_ctr, penalty_same_hinge, penalty_diff_hinge)
            print('Done Training for epoch: ', epoch)
        
            
    def train_erm_phase(self):
        
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
        
        for epoch in range(epochs):    
            
            penalty_erm=0
            penalty_erm_extra=0
            penalty_ws=0
            train_acc= 0.0
            train_size=0
    
            perm = torch.randperm(data_match_tensor.size(0))            
            data_match_tensor_split= torch.split(data_match_tensor[perm], args.batch_size, dim=0)
            label_match_tensor_split= torch.split(label_match_tensor[perm], args.batch_size, dim=0)
            print('Split Matched Data: ', len(data_match_tensor_split), data_match_tensor_split[0].shape, len(label_match_tensor_split))
    
            #Batch iteration over single epoch
            for batch_idx, (x_e, y_e ,d_e, idx_e) in enumerate(train_dataset):
        #         print('Batch Idx: ', batch_idx)

                opt.zero_grad()
                loss_e= torch.tensor(0.0).to(cuda)

                x_e= x_e.to(cuda)
                y_e= torch.argmax(y_e, dim=1).to(cuda)
                d_e= torch.argmax(d_e, dim=1).numpy()
                
                #Forward Pass
                out= phi(x_e)
                erm_loss_extra= erm_loss(out, y_e)    
                penalty_erm_extra += float(loss_extra)

                wasserstein_loss=torch.tensor(0.0).to(cuda)
                erm_loss= torch.tensor(0.0).to(cuda) 
                if epoch > anneal_iter:
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
        
                    erm_loss+= erm_loss(feat_match, label_match)
                    penalty_erm+= float(erm_loss)                
            
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

                    
                        loss_e += ( args.penalty_ws_erm*( epoch-anneal_iter )/(args.epochs_erm -anneal_iter) )*wasserstein_loss

                    loss_e += args.penalty_erm*erm_loss
                    loss_e += args.penlaty_erm*erm_loss_extra
                        
                loss_e.backward(retain_graph=False)
                opt.step()
                
                del erm_loss_extra
                del erm_loss
                del wasserstein_loss 
                del loss_e
                torch.cuda.empty_cache()
        
                train_acc+= torch.sum(torch.argmax(out, dim=1) == y_e ).item()
                train_size+= y_e.shape[0]
                
            print('Train Loss Basic : ', penalty_erm_extra,  penalty_erm, penalty_ws )
            print('Train Acc Env : ', 100*train_acc/train_size )
            print('Done Training for epoch: ', epoch)        
