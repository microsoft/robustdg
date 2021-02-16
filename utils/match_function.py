import numpy as np
import torch

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

def init_data_match_dict(args, keys, vals, variation):
    data={}
    for key in keys:
        data[key]={}
        if variation:
            val_dim= vals[key]
        else:
            val_dim= vals
        
#         if args.dataset == 'color_mnist':        
#             data[key]['data']=torch.rand((val_dim, 2, 28, 28))
#         elif args.dataset == 'rot_mnist' or args.dataset == 'fashion_mnist':
#             if args.model_name == 'lenet':            
#                 data[key]['data']=torch.rand((val_dim, 1, 32, 32))      
#             elif args.model_name == 'resnet18':
#                 data[key]['data']=torch.rand((val_dim, 1, 224, 224))      
#         elif args.dataset == 'pacs':
#             data[key]['data']=torch.rand((val_dim, 3, 227, 227))
        
        data[key]['data']= torch.rand((val_dim, args.img_c, args.img_w, args.img_h))
        data[key]['label']=torch.rand((val_dim, 1))
        data[key]['idx']=torch.randint(0, 1, (val_dim, 1))
    return data

def get_matched_pairs(args, cuda, train_dataset, domain_size, total_domains, training_list_size, phi, match_case, perfect_match, inferred_match):        
    
    #Making Data Matched pairs
    data_matched= init_data_match_dict( args, range(domain_size), total_domains, 0 )
    domain_data= init_data_match_dict( args, range(total_domains), training_list_size, 1)
    indices_matched={}
    for key in range(domain_size):
        indices_matched[key]=[]
    perfect_match_rank=[]
        
    domain_count={}
    for domain in range(total_domains):
        domain_count[domain]= 0
        
    # Create dictionary: class label -> list of ordered indices
    if args.method_name == 'hybrid':
        for batch_idx, (x_e, _, y_e ,d_e, idx_e) in enumerate(train_dataset):
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
    else:
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
            if perfect_match:
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
                    # Need to compute over batches of base_fear due to CUDA Memory out errors
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
                        if perfect_match:                        
                            if inferred_match:
                                curr_indice= ordered_curr_indices[match_idx[idx]].item()
#                                 print('Curr Indice, Idx: ', curr_indice, idx)
#                                 print('Sort, OrdIndices: ', sort_idx.shape, ordered_curr_indices.shape, type(ordered_curr_indices))
#                                 print('Perfect Indice: ', perfect_indice )
#                                 print('Unique OrdIndices: ', len(torch.unique(ordered_curr_indices[sort_idx[idx]])))
#                                 print( perfect_indice in ordered_curr_indices)
#                                 print( perfect_indice in ordered_curr_indices[sort_idx[idx]] )
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
            
        if perfect_match and inferred_match ==0 and domain_idx != base_domain_idx and  total_rand_counter < perm_size:
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
    
    print(data_match_tensor.shape, label_match_tensor.shape)
    
    del domain_data
    del data_matched

    return data_match_tensor, label_match_tensor, indices_matched, perfect_match_rank
