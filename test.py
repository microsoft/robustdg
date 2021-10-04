#Common imports
import os
import sys
import numpy as np
import argparse
import copy
import random
import json
import pickle

#Sklearn
import sklearn
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

#robustdg
from utils.helper import *
from utils.match_function import *

# Input Parsing
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='rot_mnist', 
                    help='Datasets: rot_mnist; fashion_mnist; pacs')
parser.add_argument('--method_name', type=str, default='erm_match', 
                    help=' Training Algorithm: erm_match; matchdg_ctr; matchdg_erm')
parser.add_argument('--model_name', type=str, default='resnet18', 
                    help='Architecture of the model to be trained')
parser.add_argument('--train_domains', nargs='+', type=str, default=["15", "30", "45", "60", "75"], 
                    help='List of train domains')
parser.add_argument('--test_domains', nargs='+', type=str, default=["0", "90"], 
                    help='List of test domains')
parser.add_argument('--out_classes', type=int, default=10, 
                    help='Total number of classes in the dataset')
parser.add_argument('--img_c', type=int, default= 1, 
                    help='Number of channels of the image in dataset')
parser.add_argument('--img_h', type=int, default= 224, 
                    help='Height of the image in dataset')
parser.add_argument('--img_w', type=int, default= 224, 
                    help='Width of the image in dataset')
parser.add_argument('--fc_layer', type=int, default= 1, 
                    help='ResNet architecture customization; 0: No fc_layer with resnet; 1: fc_layer for classification with resnet')
parser.add_argument('--match_layer', type=str, default='logit_match', 
                    help='rep_match: Matching at an intermediate representation level; logit_match: Matching at the logit level')
parser.add_argument('--pos_metric', type=str, default='l2', 
                    help='Cost to function to evaluate distance between two representations; Options: l1; l2; cos')
parser.add_argument('--rep_dim', type=int, default=250, 
                    help='Representation dimension for contrsative learning')
parser.add_argument('--pre_trained',type=int, default=0, 
                    help='0: No Pretrained Architecture; 1: Pretrained Architecture')
parser.add_argument('--perfect_match', type=int, default=1, 
                    help='0: No perfect match known (PACS); 1: perfect match known (MNIST)')
parser.add_argument('--opt', type=str, default='sgd', 
                    help='Optimizer Choice: sgd; adam')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                   help='Weight Decay in SGD')
parser.add_argument('--lr', type=float, default=0.01, 
                    help='Learning rate for training the model')
parser.add_argument('--batch_size', type=int, default=16, 
                    help='Batch size foe training the model')
parser.add_argument('--epochs', type=int, default=15, 
                    help='Total number of epochs for training the model')
parser.add_argument('--penalty_s', type=int, default=-1, 
                    help='Epoch threshold over which Matching Loss to be optimised')
parser.add_argument('--penalty_irm', type=float, default=0.0, 
                    help='Penalty weight for IRM invariant classifier loss')
parser.add_argument('--penalty_aug', type=float, default=1.0, 
                    help='Penalty weight for Augmentation in Hybrid approach loss')
parser.add_argument('--penalty_ws', type=float, default=0.1, 
                    help='Penalty weight for Matching Loss')
parser.add_argument('--penalty_diff_ctr',type=float, default=1.0, 
                    help='Penalty weight for Contrastive Loss')
parser.add_argument('--tau', type=float, default=0.05, 
                    help='Temperature hyper param for NTXent contrastive loss ')
parser.add_argument('--match_flag', type=int, default=0, 
                    help='0: No Update to Match Strategy; 1: Updates to Match Strategy')
parser.add_argument('--match_case', type=float, default=1.0, 
                    help='0: Random Match; 1: Perfect Match. 0.x" x% correct Match')
parser.add_argument('--match_interrupt', type=int, default=5, 
                    help='Number of epochs before inferring the match strategy')
parser.add_argument('--ctr_abl', type=int, default=0, 
                    help='0: Randomization til class level ; 1: Randomization completely')
parser.add_argument('--match_abl', type=int, default=0, 
                    help='0: Randomization til class level ; 1: Randomization completely')
parser.add_argument('--n_runs', type=int, default=3, 
                    help='Number of iterations to repeat the training process')
parser.add_argument('--n_runs_matchdg_erm', type=int, default=1, 
                    help='Number of iterations to repeat training process for matchdg_erm')
parser.add_argument('--ctr_model_name', type=str, default='resnet18', 
                    help='(For matchdg_ctr phase) Architecture of the model to be trained')
parser.add_argument('--ctr_match_layer', type=str, default='logit_match', 
                    help='(For matchdg_ctr phase) rep_match: Matching at an intermediate representation level; logit_match: Matching at the logit level')
parser.add_argument('--ctr_match_flag', type=int, default=1, 
                    help='(For matchdg_ctr phase) 0: No Update to Match Strategy; 1: Updates to Match Strategy')
parser.add_argument('--ctr_match_case', type=float, default=0.01, 
                    help='(For matchdg_ctr phase) 0: Random Match; 1: Perfect Match. 0.x" x% correct Match')
parser.add_argument('--ctr_match_interrupt', type=int, default=5, 
                    help='(For matchdg_ctr phase) Number of epochs before inferring the match strategy')
parser.add_argument('--mnist_seed', type=int, default=0, 
                    help='Change it between 0-6 for different subsets of Mnist and Fashion Mnist dataset')
parser.add_argument('--retain', type=float, default=0, 
                    help='0: Train from scratch in MatchDG Phase 2; 2: Finetune from MatchDG Phase 1 in MatchDG is Phase 2')
parser.add_argument('--cuda_device', type=int, default=0, 
                    help='Select the cuda device by id among the avaliable devices' )
parser.add_argument('--os_env', type=int, default=0, 
                    help='0: Code execution on local server/machine; 1: Code execution in docker/clusters' )


#Differential Privacy
parser.add_argument('--dp_noise', type=int, default=0, 
                    help='0: No DP noise; 1: Add DP noise')
parser.add_argument('--dp_epsilon', type=float, default=1.0, 
                    help='Epsilon value for Differential Privacy')
parser.add_argument('--dp_attach_opt', type=int, default=1, 
                    help='0: Infinite Epsilon; 1: Finite Epsilion')



#MMD, DANN
parser.add_argument('--d_steps_per_g_step', type=int, default=1)
parser.add_argument('--grad_penalty', type=float, default=0.0)
parser.add_argument('--conditional', type=int, default=1)
parser.add_argument('--gaussian', type=int, default=1)


#Slab Dataset
parser.add_argument('--slab_data_dim', type=int, default= 2, 
                    help='Number of features in the slab dataset')
parser.add_argument('--slab_total_slabs', type=int, default=7)
parser.add_argument('--slab_num_samples', type=int, default=1000)
parser.add_argument('--slab_noise', type=float, default=0.1)


#Differentiate between resnet, lenet, domainbed cases of mnist
parser.add_argument('--mnist_case', type=str, default='resnet18', 
                    help='MNIST Dataset Case: resnet18; lenet, domainbed')
parser.add_argument('--mnist_aug', type=int, default=0, 
                    help='MNIST Data Augmentation: 0 (MNIST, FMNIST Privacy Evaluation); 1 (FMNIST)')
 
    
#Multiple random matches
parser.add_argument('--total_matches_per_point', type=int, default=1, 
                    help='Multiple random matches')


# Evaluation specific
parser.add_argument('--test_metric', type=str, default='acc', 
                    help='Evaluation Metrics: acc; match_score, t_sne, mia')
parser.add_argument('--acc_data_case', type=str, default='test', 
                    help='Dataset Train/Val/Test for the accuracy evaluation metric')
parser.add_argument('--top_k', type=int, default=10, 
                    help='Top K matches to consider for the match score evaluation metric')
parser.add_argument('--match_func_aug_case', type=int, default=0, 
                    help='0: Evaluate match func on train domains; 1: Evaluate match func on self augmentations')
parser.add_argument('--match_func_data_case', type=str, default='train', 
                    help='Dataset Train/Val/Test for the match score evaluation metric')
parser.add_argument('--mia_batch_size', default=64, type=int, 
                    help='batch size')
parser.add_argument('--mia_dnn_steps', default=5000, type=int,
                    help='number of training steps')
parser.add_argument('--mia_sample_size', default=1000, type=int,
                    help='number of samples from train/test dataset logits')
parser.add_argument('--mia_logit', default=1, type=int,
                    help='0: Softmax applied to logits; 1: No Softmax applied to logits')
parser.add_argument('--attribute_domain', default=1, type=int, 
                   help='0: spur correlations as attribute; 1: domain as attribute')
parser.add_argument('--adv_eps', default=0.3, type=float,
                    help='Epsilon ball dimension for PGD attacks')
parser.add_argument('--logit_plot_path', default='', type=str,
                    help='File name to save logit/loss plots')

args = parser.parse_args()

#GPU
cuda= torch.device("cuda:" + str(args.cuda_device))
if cuda:
    kwargs = {'num_workers': 1, 'pin_memory': False} 
else:
    kwargs= {}

args.kwargs= kwargs
    
#List of Train; Test domains
train_domains= args.train_domains
test_domains= args.test_domains

#Initialize
final_metric_score=[]

res_dir= 'results/'
if args.dp_noise:
    base_res_dir=(
                res_dir + args.dataset_name + '/' + 'dp_' +  str(args.dp_epsilon) + '_' + args.method_name + '/' + args.match_layer 
                + '/' + 'train_' + str(args.train_domains)
            )    
else:
    base_res_dir=(
                res_dir + args.dataset_name + '/' + args.method_name + '/' + args.match_layer 
                + '/' + 'train_' + str(args.train_domains)
            )

print('Result Base Dir: ', base_res_dir)

#TODO: Handle slab noise case in helper functions
if args.dataset_name == 'slab':
    base_res_dir= base_res_dir + '/slab_noise_'  + str(args.slab_noise)

if not os.path.exists(base_res_dir):
    os.makedirs(base_res_dir)    

#Checks
if args.method_name == 'matchdg_ctr' and args.test_metric == 'acc':
    raise ValueError('Match DG during the contrastive learning phase cannot be evaluted for test accuracy metric')
    sys.exit()

if args.perfect_match == 0 and args.test_metric == 'match_score' and args.match_func_aug_case==0:
    raise ValueError('Cannot evalute match function metrics when perfect match is not known')
    sys.exit()
    
#Execute the method for multiple runs ( total args.n_runs )
for run in range(args.n_runs):
    
    #Seed for repoduability
    random.seed(run*10) 
    np.random.seed(run*10) 
    torch.manual_seed(run*10)    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(run*10)    
    
    #DataLoader        
    train_dataset= torch.empty(0)
    val_dataset= torch.empty(0)
    test_dataset= torch.empty(0)
    if args.test_metric in ['match_score', 'feat_eval', 'slab_feat_eval']:
        if args.match_func_data_case== 'train':
            train_dataset= get_dataloader( args, run, train_domains, 'train', 1, kwargs )
        elif args.match_func_data_case== 'val':
            val_dataset= get_dataloader( args, run, train_domains, 'val', 1, kwargs )
        elif args.match_func_data_case== 'test':
            test_dataset= get_dataloader( args, run, test_domains, 'test', 1, kwargs )
    elif args.test_metric in ['acc', 'per_domain_acc']:
        if args.acc_data_case== 'train':
            train_dataset= get_dataloader( args, run, train_domains, 'train', 1, kwargs )
        elif args.acc_data_case== 'val':
            val_dataset= get_dataloader( args, run, train_domains, 'val', 1, kwargs )
        elif args.acc_data_case== 'test':
            test_dataset= get_dataloader( args, run, test_domains, 'test', 1, kwargs )
    elif args.test_metric in ['mia', 'privacy_entropy', 'privacy_loss_attack']:
        train_dataset= get_dataloader( args, run, train_domains, 'train', 1, kwargs )
        test_dataset= get_dataloader( args, run, test_domains, 'test', 1, kwargs )
    elif args.test_metric == 'attribute_attack':
        print( train_domains + test_domains)
        train_dataset= get_dataloader( args, run, train_domains + test_domains, 'train', 1, kwargs )
        test_dataset= get_dataloader( args, run, train_domains + test_domains, 'test', 1, kwargs )        
    else:
        test_dataset= get_dataloader( args, run, test_domains, 'test', 1, kwargs )
        
#     print('Train Domains, Domain Size, BaseDomainIdx, Total Domains: ', train_domains, total_domains, domain_size, training_list_size)
    
    #Import the testing module
    if args.test_metric == 'acc':
        from evaluation.base_eval import BaseEval
        test_method= BaseEval(
                              args, train_dataset, val_dataset,
                              test_dataset, base_res_dir,
                              run, cuda
                             )

    elif args.test_metric == 'per_domain_acc':
        from evaluation.per_domain_acc import PerDomainAcc
        test_method= PerDomainAcc(
                              args, train_dataset, val_dataset,
                              test_dataset, base_res_dir,
                              run, cuda
                             )
        
    elif args.test_metric == 'match_score':
        from evaluation.match_eval import MatchEval
        test_method= MatchEval(
                               args, train_dataset, val_dataset,
                               test_dataset, base_res_dir, 
                               run, cuda
                              )   

    elif args.test_metric == 'feat_eval':
        from evaluation.feat_eval import FeatEval
        test_method= FeatEval(
                               args, train_dataset, val_dataset,
                               test_dataset, base_res_dir, 
                               run, cuda
                              )   
        
    elif args.test_metric == 'slab_feat_eval':
        from evaluation.slab_feat_eval import SlabFeatEval
        test_method= SlabFeatEval(
                               args, train_dataset, val_dataset,
                               test_dataset, base_res_dir, 
                               run, cuda
                              )         

    elif args.test_metric == 't_sne':
        from evaluation.t_sne import TSNE
        test_method= TSNE(
                              args, train_dataset, val_dataset,
                              test_dataset, base_res_dir,
                              run, cuda
                             )        
        
    elif args.test_metric == 'mia':
        from evaluation.privacy_attack import PrivacyAttack
        test_method= PrivacyAttack(
                              args, train_dataset, val_dataset,
                              test_dataset, base_res_dir,
                              run, cuda
                             )       
        
    elif args.test_metric == 'attribute_attack':
        from evaluation.attribute_attack import AttributeAttack
        test_method= AttributeAttack(
                              args, train_dataset, val_dataset,
                              test_dataset, base_res_dir,
                              run, cuda
                             )        

    elif args.test_metric == 'privacy_loss_attack':
        from evaluation.privacy_loss_attack import PrivacyLossAttack
        test_method= PrivacyLossAttack(
                              args, train_dataset, val_dataset,
                              test_dataset, base_res_dir,
                              run, cuda
                             )        
        
    elif args.test_metric == 'privacy_entropy':
        from evaluation.privacy_entropy import PrivacyEntropy
        test_method= PrivacyEntropy(
                              args, train_dataset, val_dataset,
                              test_dataset, base_res_dir,
                              run, cuda
                             )        

    elif args.test_metric == 'logit_hist':
        from evaluation.logit_hist import LogitHist
        test_method= LogitHist(
                              args, train_dataset, val_dataset,
                              test_dataset, base_res_dir,
                              run, cuda
                             )        
        
    elif args.test_metric == 'adv_attack':
        from evaluation.adv_attack import AdvAttack
        test_method= AdvAttack(
                              args, train_dataset, val_dataset,
                              test_dataset, base_res_dir,
                              run, cuda
                             )        
        
    #Testing Phase
    with torch.no_grad():
        if args.test_metric == 'mia':
            for mia_run in range(2):
                if args.method_name in ['matchdg_erm', 'hybrid']:
                    for run_matchdg_erm in range(args.n_runs_matchdg_erm):   
                        test_method.get_model(run_matchdg_erm)        
                        test_method.get_metric_eval()
                        final_metric_score.append( test_method.metric_score )
                else:
                    test_method.get_model()        
                    test_method.get_metric_eval()
                    final_metric_score.append( test_method.metric_score )
        else:
            if args.method_name in ['matchdg_erm', 'hybrid']:
                for run_matchdg_erm in range(args.n_runs_matchdg_erm):   
                    test_method.get_model(run_matchdg_erm)        
                    test_method.get_metric_eval()
                    final_metric_score.append( test_method.metric_score )
            else:
                test_method.get_model()        
                test_method.get_metric_eval()
                final_metric_score.append( test_method.metric_score )    

if args.test_metric not in ['t_sne', 'logit_hist']:
    print('\n')
    print('Done for Model..')

    keys=final_metric_score[0].keys()
    for key in keys:
        curr_metric_score=[]
        for item in final_metric_score:
            curr_metric_score.append( item[key] )
        curr_metric_score= np.array(curr_metric_score)
        print(key, ' : ', np.mean(curr_metric_score), np.std(curr_metric_score)/np.sqrt(curr_metric_score.shape[0]))

    print('\n')
