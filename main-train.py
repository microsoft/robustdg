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
    
def embedding_dist(x1, x2, xent=False):
    
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
        cos_sim= cos_sim / args.tau
        
        if torch.sum( torch.isnan( cos_sim ) ):
            print('Cos is nan')
            sys.exit()
                
        loss= torch.sum( torch.exp(cos_sim), dim=1)
        
        if torch.sum( torch.isnan( loss ) ):
            print('Loss is nan')
            sys.exit()
        
        return loss
        
    else:    
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
            if args.model_name == 'lenet':            
                data[key]['data']=torch.rand((val_dim, 1, 32, 32))      
            else:
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
    for domain in range(total_domains):
        if domain_count[domain] != training_list_size[domain]:
            print('Issue: Some data points are missing from domain_data dictionary')
    
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
        print('Base Domain: ', base_domain_size, base_domain_idx, y_c )    
    
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
            if args.perfect_match:
                if not torch.equal(ordered_base_indices, ordered_curr_indices):
                    print('Issue: Different indices across domains for perfect match' )
            
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
        
        if total_data_idx != domain_size:
            print('Issue: Some data points left from data_matched dictionary', total_data_idx, domain_size)
            
        if args.perfect_match and inferred_match ==0 and domain_idx != base_domain_idx and  total_rand_counter < perm_size:
            print('Issue: Total random changes made are less than perm_size for domain', domain_idx, total_rand_counter, perm_size)
                            
                
    # Sanity Check:  N keys; K vals per key
    for key in data_matched.keys():
        if data_matched[key]['label'].shape[0] != total_domains:
            print('Issue with data matching')

    #Sanity Check: Ensure paired points have the same class label
    wrong_case=0
    for key in data_matched.keys():
        for d_i in range(data_matched[key]['label'].shape[0]):
            for d_j in range(data_matched[key]['label'].shape[0]):
                if d_j > d_i:
                    if data_matched[key]['label'][d_i] != data_matched[key]['label'][d_j]:
                        wrong_case+=1
    print('Total Label MisMatch across pairs: ', wrong_case )
            
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
    
    print(data_match_tensor.shape, label_match_tensor.shape)
    
    del domain_data
    del data_matched

    return data_match_tensor, label_match_tensor, indices_matched, perfect_match_rank
    
        
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
                            
                        neg_dist= embedding_dist( pos_feat_match[:, d_i, :], diff_neg_feat_match[:, :], xent=True)     
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
                        
                '''
                total_same_neg_exp=1
                total_diff_neg_exp=25
                same_neg_counter=0
                diff_neg_counter=0


                for y_c in range(args.out_classes):

                    pos_indices= label_match[:,0] == y_c
                    neg_indices= label_match[:,0] != y_c
                    pos_feat_match= feat_match[pos_indices]
                    neg_feat_match= feat_match[neg_indices]
                    
                    if pos_feat_match.shape[0] > neg_feat_match.shape[0]:
                        print('Weird! Positive Matches are more than the negative matches?', pos_feat_match.shape[0], neg_feat_match.shape[0])
                     
                    # If no instances of label y_c in the current batch then continue
                    if pos_feat_match.shape[0] ==0 or neg_feat_match.shape[0] == 0:
                        continue
                        
#                     if batch_idx==1:
#                         print('Pos Feat Match: ', pos_feat_match.shape)

                    # Iterating over anchors from different domains
                    for d_i in range(pos_feat_match.shape[1]):
                        # Removing the criteria d_i == base_domain_idx
    #                     if d_i != base_domain_idx:
    #                         continue 

    #                     #Iterating over positive matches from different domains
    #                     for d_j in range(pos_feat_match.shape[1]):
    #                         # Removing the d_j > d_i constraint to include all different cases if same class pairs
    #                         if d_i == d_j:
#                                 continue                            

                        # Average pos dist for current anchor
                        pos_dist_list= [] 
                        for d_j in range(pos_feat_match.shape[1]):
                            if d_i != d_j:
                                pos_dist= embedding_dist( pos_feat_match[:, d_i, :], pos_feat_match[:, d_j, :] )
                                pos_dist_list.append(pos_dist)
                        #
                        
#                         # Average pos dist for current anchor
#                         pos_dist= torch.zeros((pos_feat_match.size(0))).to(cuda)
#                         for d_j in range(pos_feat_match.shape[1]):
#                             if d_i != d_j:
#                                 pos_dist+= embedding_dist( pos_feat_match[:, d_i, :], pos_feat_match[:, d_j, :] )
#                         pos_dist= pos_dist/(pos_feat_match.shape[1]-1)

#                         # Average neg dist for current anchor
#                         neg_dist=torch.zeros((pos_feat_match.size(0))).to(cuda)
#                         for neg_idx in range(total_same_neg_exp):
#                             perm= torch.randperm(pos_feat_match.size(0))
#                             same_neg_feat_match= pos_feat_match[perm]
#                             for d_j in range(pos_feat_match.shape[1]):                            
#                                 neg_dist+= embedding_dist( pos_feat_match[:, d_i, :], same_neg_feat_match[:, d_j, :]) 
#                         neg_dist= neg_dist / (total_same_neg_exp*pos_feat_match.shape[1])


#                         same_hinge_loss+=  F.hinge_embedding_loss( neg_dist - pos_dist, torch.tensor(-1).to(cuda), args.same_margin, reduction='sum').to(cuda)  
#                         same_ctr_loss+= torch.sum(neg_dist)
#                         same_neg_counter+= pos_feat_match.shape[0]
#                         del neg_dist    

                        # Average neg dist for current anchor
                        neg_dist=torch.zeros((pos_feat_match.size(0))).to(cuda)
                        for neg_idx in range(total_diff_neg_exp):                                          
                            perm= torch.randperm(neg_feat_match.size(0))
                            diff_neg_feat_match= neg_feat_match[perm]
                            
                            # To ensure we have a negative match for each anchor
                            total_anchors= pos_feat_match.shape[0]
                            total_neg= diff_neg_feat_match.shape[0]
                            if total_anchors > total_neg:
                                extra_num= total_anchors - total_neg
                                extra_neg= diff_neg_feat_match[:extra_num]
                                diff_neg_feat_match= torch.cat( (diff_neg_feat_match, extra_neg) )
                            else:
                                diff_neg_feat_match= diff_neg_feat_match[:total_anchors]
                            
                            for d_j in range(pos_feat_match.shape[1]):
#                                 neg_dist+= embedding_dist( pos_feat_match[:, d_i, :], diff_neg_feat_match[:, d_j, :])                                
                                neg_dist= embedding_dist( pos_feat_match[:, d_i, :], diff_neg_feat_match[:, d_j, :])                                
                                for pos_dist in pos_dist_list:
                                    diff_hinge_loss+=  F.hinge_embedding_loss( neg_dist - pos_dist, torch.tensor(-1).to(cuda), args.diff_margin, reduction='sum').to(cuda) 
                                    diff_neg_counter+= pos_feat_match.shape[0]
                                
#                         neg_dist= neg_dist / (total_diff_neg_exp*pos_feat_match.shape[1])                       

                        #print('Diff Hinge Comp ', torch.mean(pos_dist), torch.mean(neg_dist))
#                         if d_i ==0 and batch_idx ==1 :
#                             print( torch.mean(neg_dist), torch.mean(pos_dist), neg_dist.shape, pos_dist.shape )
#                             print( neg_dist[0], pos_dist[0],  F.hinge_embedding_loss( neg_dist[0] - pos_dist[0], torch.tensor(-1).to(cuda), args.diff_margin, reduction='sum').to(cuda) )

#                         for pos_dist in pos_dist_list:
#                             diff_hinge_loss+=  F.hinge_embedding_loss( neg_dist - pos_dist, torch.tensor(-1).to(cuda), args.diff_margin, reduction='sum').to(cuda)  
#                             diff_neg_counter+= pos_feat_match.shape[0]
                            
                        diff_ctr_loss+= torch.sum(neg_dist)
                        
                        del neg_dist                        
                        del pos_dist

                    del pos_indices
                    del neg_indices
                    del pos_feat_match
                    del neg_feat_match
                
                '''
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
        
    print('T-SNE', np.array(t_sne_out).shape, feat_all.shape, label_all.shape, domain_all.shape)
        
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
                        data_match_tensor, label_match_tensor, indices_matched, perfect_match_rank= get_matched_pairs( train_dataset, domain_size, total_domains, training_list_size, phi, args.match_case, inferred_match )
                        match_counter+=1
                        #Reset the weights after very match strategy update
        #                 phi= RotMNIST( feature_dim, num_classes ).to(cuda)
        #                 opt= optim.Adam([
        #                         {'params': filter(lambda p: p.requires_grad, phi.predict_conv_net.parameters()) },
        #                         {'params': filter(lambda p: p.requires_grad, phi.predict_fc_net.parameters()) },
        #                         {'params': filter(lambda p: p.requires_grad, phi.predict_final_net.parameters()) }                   
        #                     ], lr=learning_rate)                

                    elif args.match_flag ==0 or match_counter>=1:
                        temp_1, temp_2, indices_matched, perfect_match_rank= get_matched_pairs( train_dataset, domain_size, total_domains, training_list_size, phi, args.match_case, inferred_match )                

                    perfect_match_rank= np.array(perfect_match_rank)
                    if args.perfect_match:
                        print('Mean Perfect Match Score: ', np.mean(perfect_match_rank), 100*np.sum(perfect_match_rank < 10)/perfect_match_rank.shape[0] )
                        match_rank.append( np.mean(perfect_match_rank) )
                        match_top_k.append( 100*np.sum( perfect_match_rank < 10 )/perfect_match_rank.shape[0] )

                else:
                    inferred_match=0
                    data_match_tensor, label_match_tensor, indices_matched, perfect_match_rank= get_matched_pairs( train_dataset, domain_size, total_domains, training_list_size, phi, args.match_case, inferred_match )

                ## To ensure a random match keeps happening after every match interrupt
    #             data_match_tensor, label_match_tensor, indices_matched= get_matched_pairs( train_dataset, domain_size, total_domains, base_domain_idx, args.match_case )          
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

        #T-SNE
#         if args.ctr_phase:
#             t_sne_out= compute_t_sne( train_dataset, phi )
#             with open(base_res_dir + args.method_name + sub_dir + '/TSNE_' + post_string + '.json', 'w') as fp:
#                 json.dump(t_sne_out, fp)                        
    
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
                data_match_tensor, label_match_tensor, indices_matched, perfect_match_rank= get_matched_pairs( train_dataset, domain_size, total_domains, training_list_size, phi, args.match_case_erm, inferred_match )                

                if args.perfect_match:
                    score= perfect_match_score(indices_matched)
                    print('Perfect Match Score: ', score)                    
                    perfect_match_rank= np.array(perfect_match_rank)            
                    print('Mean Perfect Match Score: ', np.mean(perfect_match_rank), 100*np.sum(perfect_match_rank < 10)/perfect_match_rank.shape[0] )

            else:
                inferred_match=0
                # x% percentage match initial strategy
                data_match_tensor, label_match_tensor, indices_matched, perfect_match_rank= get_matched_pairs( train_dataset, domain_size, total_domains, training_list_size, phi, args.match_case_erm, inferred_match )                
               
                if args.perfect_match:
                    score= perfect_match_score(indices_matched)
                    print('Perfect Match Score: ', score)                    
                    perfect_match_rank= np.array(perfect_match_rank)            
                    print('Mean Perfect Match Score: ', np.mean(perfect_match_rank), 100*np.sum(perfect_match_rank < 10)/perfect_match_rank.shape[0] )

                    
            # Model and paramters
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