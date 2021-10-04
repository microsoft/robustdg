import sys
import numpy as np
import argparse
import copy
import random
import json
import os
from more_itertools import chunked

import torch
from torch.autograd import grad
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import torch.utils.data as data_utils


from utils.match_function import get_matched_pairs

print('From Inside the Algo Class: ', sys.argv[0])


def get_noise_multiplier(
    target_epsilon: float,
    target_delta: float,
    sample_rate: float,
    epochs: int,
    alphas: [float],
    sigma_min: float = 0.01,
    sigma_max: float = 10.0,
) -> float:
    r"""
    Computes the noise level sigma to reach a total budget of (target_epsilon, target_delta)
    at the end of epochs, with a given sample_rate

    Args:
        target_epsilon: the privacy budget's epsilon
        target_delta: the privacy budget's delta
        sample_rate: the sampling rate (usually batch_size / n_data)
        epochs: the number of epochs to run
        alphas: the list of orders at which to compute RDP

    Returns:
        The noise level sigma to ensure privacy budget of (target_epsilon, target_delta)

    """
    
    from opacus import privacy_analysis
    
    eps = float("inf")
    while eps > target_epsilon:
        sigma_max = 2 * sigma_max
        rdp = privacy_analysis.compute_rdp(
            sample_rate, sigma_max, epochs / sample_rate, alphas
        )
        eps = privacy_analysis.get_privacy_spent(alphas, rdp, target_delta)[0]
        if sigma_max > 2000:
            raise ValueError("The privacy budget is too low.")

    while sigma_max - sigma_min > 0.01:
        sigma = (sigma_min + sigma_max) / 2
        rdp = privacy_analysis.compute_rdp(
            sample_rate, sigma, epochs / sample_rate, alphas
        )
        eps = privacy_analysis.get_privacy_spent(alphas, rdp, target_delta)[0]

        if eps < target_epsilon:
            sigma_max = sigma
        else:
            sigma_min = sigma

    return sigma


class BaseAlgo():
    def __init__(self, args, train_dataset, val_dataset, test_dataset, base_res_dir, run, cuda):
        
        
        self.args= args
        self.train_dataset= train_dataset['data_loader']
        if args.method_name == 'matchdg_ctr':
            self.val_dataset= val_dataset
        else:
            self.val_dataset= val_dataset['data_loader']
        self.test_dataset= test_dataset['data_loader']
        
        self.train_domains= train_dataset['domain_list']
        self.total_domains= train_dataset['total_domains']
        self.domain_size= train_dataset['base_domain_size'] 
        self.training_list_size= train_dataset['domain_size_list']
        
        self.base_res_dir= base_res_dir
        self.run= run
        self.cuda= cuda
        
        self.post_string= str(self.args.penalty_ws) + '_' + str(self.args.penalty_diff_ctr) + '_' + str(self.args.match_case) + '_' + str(self.args.match_interrupt) + '_' + str(self.args.match_flag) + '_' + str(self.run) + '_' + self.args.pos_metric + '_' + self.args.model_name
        
        self.phi= self.get_model()
        self.opt= self.get_opt()
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt, step_size=25)    
        
        self.final_acc=[]
        self.val_acc=[]
        self.train_acc=[]
        
        # Differentially Private Noise
        if self.args.dp_noise:
            self.privacy_engine= self.get_dp_noise()
    
    def get_model(self):
        
        if self.args.model_name == 'lenet':
            from models.lenet import LeNet5
            phi= LeNet5()

        if self.args.model_name == 'slab':
            from models.slab import SlabClf
            if self.args.method_name in ['csd', 'matchdg_ctr']:
                fc_layer=0
            else:
                fc_layer= self.args.fc_layer
            phi= SlabClf(self.args.slab_data_dim, self.args.out_classes, fc_layer)
            
        if self.args.model_name == 'fc':
            from models.fc import FC
            if self.args.method_name in ['csd', 'matchdg_ctr']:
                fc_layer=0
            else:
                fc_layer= self.args.fc_layer
            phi= FC(self.args.out_classes, fc_layer)
            
        if self.args.model_name == 'domain_bed_mnist':
            from models.domain_bed_mnist import DomainBed
            if self.args.method_name in ['csd', 'matchdg_ctr']:
                fc_layer=0
            else:
                fc_layer= self.args.fc_layer                        
            phi= DomainBed(self.args.img_c, fc_layer)
            
        if self.args.model_name == 'alexnet':
            from models.alexnet import alexnet
            if self.args.method_name in ['csd', 'matchdg_ctr']:
                fc_layer=0
            else:
                fc_layer= self.args.fc_layer            
            phi= alexnet(self.args.model_name, self.args.out_classes, fc_layer, 
                            self.args.img_c, self.args.pre_trained, self.args.os_env)
            
        if 'resnet' in self.args.model_name:
            from models.resnet import get_resnet
            if self.args.method_name in ['csd', 'matchdg_ctr']:
                fc_layer=0
            else:
                fc_layer= self.args.fc_layer
            phi= get_resnet(self.args.model_name, self.args.out_classes, fc_layer, 
                            self.args.img_c, self.args.pre_trained, self.args.dp_noise, self.args.os_env)
            
        if 'densenet' in self.args.model_name:
            from models.densenet import get_densenet
            if self.args.method_name in ['csd', 'matchdg_ctr']:
                fc_layer=0
            else:
                fc_layer= self.args.fc_layer
            phi= get_densenet(self.args.model_name, self.args.out_classes, fc_layer, 
                            self.args.img_c, self.args.pre_trained, self.args.os_env)
            
        print('Model Architecture: ', self.args.model_name)
        phi=phi.to(self.cuda)
        return phi
    
    def save_model(self):
        # Store the weights of the model
        torch.save(self.phi.state_dict(), self.base_res_dir + '/Model_' + self.post_string + '.pth')
        
        # Store the validation, test loss over the training epochs
        np.save( self.base_res_dir + '/Val_Acc_' + self.post_string + '.npy', np.array(self.val_acc) )
        np.save( self.base_res_dir + '/Test_Acc_' + self.post_string + '.npy', np.array(self.final_acc))
    
    def get_opt(self):
        if self.args.opt == 'sgd':
            opt= optim.SGD([
                         {'params': filter(lambda p: p.requires_grad, self.phi.parameters()) }, 
                ], lr= self.args.lr, weight_decay= self.args.weight_decay, momentum= 0.9,  nesterov=True )        
        elif self.args.opt == 'adam':
            opt= optim.Adam([
                        {'params': filter(lambda p: p.requires_grad, self.phi.parameters())},
                ], lr= self.args.lr)
        
        return opt

    
    def get_match_function(self, inferred_match, phi):
        
        data_matched, domain_data, _= get_matched_pairs( self.args, self.cuda, self.train_dataset, self.domain_size, self.total_domains, self.training_list_size, phi, self.args.match_case, self.args.perfect_match, inferred_match )
                
#         #Start initially with randomly defined batch; else find the local approximate batch
#         if epoch > 0:                    
#             inferred_match=1
#             if self.args.match_flag:
#                 data_matched, domain_data, _= get_matched_pairs( self.args, self.cuda, self.train_dataset, self.domain_size, self.total_domains, self.training_list_size, self.phi, self.args.match_case, perfect_match, inferred_match )
#             else:
#                 temp_1, temp_2, _= get_matched_pairs( self.args, self.cuda, self.train_dataset, self.domain_size, self.total_domains, self.training_list_size, self.phi, self.args.match_case, perfect_match, inferred_match )                
#         else:
#             inferred_match=0
#             data_matched, domain_data, _= get_matched_pairs( self.args, self.cuda, self.train_dataset, self.domain_size, self.total_domains, self.training_list_size, self.phi, self.args.match_case, perfect_match, inferred_match )
        
        
        # Randomly Shuffle the list of matched data indices and divide as per batch sizes
        random.shuffle(data_matched)
        data_matched= list(chunked(data_matched, self.args.batch_size))
        
        return data_matched, domain_data

    def get_match_function_batch(self, batch_idx):
        curr_data_matched= self.data_matched[batch_idx]
        curr_batch_size= len(curr_data_matched)

        data_match_tensor=[]
        label_match_tensor=[]
        for idx in range(curr_batch_size):
            data_temp=[]
            label_temp= []
            for d_i in range(len(curr_data_matched[idx])):
                key= random.choice( curr_data_matched[idx][d_i] )
                data_temp.append(self.domain_data[d_i]['data'][key])
                label_temp.append(self.domain_data[d_i]['label'][key])
            
            data_match_tensor.append( torch.stack(data_temp) )
            label_match_tensor.append( torch.stack(label_temp) )                    

        data_match_tensor= torch.stack( data_match_tensor ) 
        label_match_tensor= torch.stack( label_match_tensor )
#         print('Shape: ', data_match_tensor.shape, label_match_tensor.shape)
        
        return data_match_tensor, label_match_tensor, curr_batch_size
    
    def get_test_accuracy(self, case):
        import opacus
        
        if self.args.dp_noise:
            opacus.autograd_grad_sample.disable_hooks()
            #self.privacy_engine.module.disable_hooks()
        
        #Test Env Code
        test_acc= 0.0
        test_size=0
        if case == 'val':
            dataset= self.val_dataset
        elif case == 'test':
            dataset= self.test_dataset

        for batch_idx, (x_e, y_e ,d_e, idx_e, obj_e) in enumerate(dataset):
            with torch.no_grad():
                
                self.opt.zero_grad()
#                 print(x_e.shape)
#                 print(torch.cuda.memory_allocated())                
                x_e= x_e.to(self.cuda)
                y_e= torch.argmax(y_e, dim=1).to(self.cuda)

                #Forward Pass
                out= self.phi(x_e)                
                
                test_acc+= torch.sum( torch.argmax(out, dim=1) == y_e ).item()
                test_size+= y_e.shape[0]
                
                # To avoid CUDA memory issues
                if self.args.dp_noise:
                    self.opt.zero_grad()

        print(' Accuracy: ', case, 100*test_acc/test_size )         
                
        #self.privacy_engine.module.enable_hooks()
        opacus.autograd_grad_sample.enable_hooks()        
        return 100*test_acc/test_size
    
    def get_dp_noise(self):
        
        print('Privacy Engine')
        print('Total Domains: ', self.total_domains, ' Domain Size ', self.domain_size, ' Batch Size ', self.args.batch_size)
        
        from opacus.dp_model_inspector import DPModelInspector
        from opacus.utils import module_modification
        
        inspector = DPModelInspector()        
#         self.phi = module_modification.convert_batchnorm_modules(self.phi) 
        inspector.validate(self.phi)
        
        MAX_GRAD_NORM = 5.0
        DELTA = 1.0/(self.total_domains*self.domain_size)
        BATCH_SIZE = self.args.batch_size * self.total_domains
        SAMPLE_RATE = BATCH_SIZE /(self.total_domains*self.domain_size)
        DEFAULT_ALPHAS = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
                
        NOISE_MULTIPLIER = get_noise_multiplier(self.args.dp_epsilon, DELTA, SAMPLE_RATE, self.args.epochs, DEFAULT_ALPHAS)
        
        print("Target Epsilon: ", self.args.dp_epsilon)
        print(f"Using sigma={NOISE_MULTIPLIER} and C={MAX_GRAD_NORM}")
        
                
        from opacus import PrivacyEngine        
        privacy_engine = PrivacyEngine(
            self.phi,
            batch_size= BATCH_SIZE,
            sample_size= self.total_domains*self.domain_size,
            noise_multiplier=NOISE_MULTIPLIER,
            max_grad_norm=MAX_GRAD_NORM,
        )
        
        if self.args.dp_attach_opt:
            print('Standard DP Training with finite epsilon')
            privacy_engine.attach(self.opt)
        else:
            print('DP Training with infinite epsilon')
            
        return privacy_engine