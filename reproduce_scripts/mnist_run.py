import os
import sys
import argparse

# Input Parsing
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='rot_mnist', 
                    help='Datasets: rot_mnist; fashion_mnist; rot_mnist_spur')
parser.add_argument('--train_case', type=str, default='train_all', 
                    help='train_all: Train with all the 5 domains; train_abl_3: Train with 3 domains; train_abl_2: Train with 2 domains')
parser.add_argument('--metric', type=str, default='train', 
                    help='train: Train the models; acc: Evaluate the train/test accuracy; privacy_loss_attack: Evaluate the MI attack robustness; match_score: Evaluate the match function statistics; attribute_attack: Evalute the attribute attack robustness')
parser.add_argument('--data_case', type=str, default='test', 
                   help='train: Evaluate the acc/match_score metrics on the train dataset; test: Evaluate the acc/match_score metrics on the test dataset')
parser.add_argument('--data_aug', type=int, default=1, 
                   help='0: No data augmentation for fashion mnist; 1: Data augmentation for fashion mnist')

parser.add_argument('--methods', nargs='+', type=str, default=['erm', 'irm', 'csd', 'rand', 'perf', 'matchdg'], 
                    help='List of methods: erm, irm, csd, rand, approx_25, approx_50, approx_75, perf, matchdg')


parser.add_argument('--dp_noise', type=int, default=0, 
                    help='0: No DP noise; 1: Add DP noise')
parser.add_argument('--dp_epsilon', type=float, default=100.0, 
                    help='Epsilon value for Differential Privacy')
parser.add_argument('--dp_attach_opt', type=int, default=1, 
                    help='0: Infinite Epsilon; 1: Finite Epsilion')

args = parser.parse_args()


dataset= args.dataset
train_case= args.train_case
metric= args.metric
data_case= args.data_case
data_aug= args.data_aug
methods= args.methods

# test_diff, test_common
test_case=['test_diff']
    
if metric == 'train':
    if dataset in ['rot_mnist', 'rot_mnist_spur']:
        base_script= 'python train.py --dataset ' + str(dataset)
    elif dataset in ['fashion_mnist']:
        base_script= 'python train.py --dataset ' + str(dataset) + ' --mnist_aug ' + str(data_aug)
    res_dir= 'results/' + str(dataset) + '/train_logs' + '/'    

elif metric == 'mia':
    if dataset in ['rot_mnist', 'rot_mnist_spur']:
        base_script= 'python  test.py --test_metric mia --mia_logit 1 --mia_sample_size 2000 --batch_size 64 ' + ' --dataset ' + str(dataset)
        base_script= 'python  test.py --test_metric mia --mia_logit 1 --mia_sample_size 2000 --batch_size 64 ' + ' --dataset ' + str(dataset)

    res_dir= 'results/'+str(dataset)+'/privacy_clf/'

elif metric == 'privacy_entropy':
    if dataset in ['rot_mnist', 'rot_mnist_spur']:
        base_script= 'python  test.py --test_metric privacy_entropy --mia_sample_size 2000 --batch_size 64 ' + ' --dataset ' + str(dataset)
    elif dataset in ['fashion_mnist']:
        base_script= 'python  test.py --test_metric privacy_entropy --mia_sample_size 2000 --batch_size 64 ' + ' --dataset ' + str(dataset)

    res_dir= 'results/'+str(dataset)+'/privacy_entropy/'

elif metric == 'privacy_loss_attack':
    if dataset in ['rot_mnist', 'rot_mnist_spur']:
        base_script= 'python  test.py --test_metric privacy_loss_attack --mia_sample_size 2000 --batch_size 64 ' + ' --dataset ' + str(dataset)
    elif dataset in ['fashion_mnist']:
        base_script= 'python  test.py --test_metric privacy_loss_attack --mia_sample_size 2000 --batch_size 64 ' + ' --dataset ' + str(dataset)

    res_dir= 'results/'+str(dataset)+'/privacy_loss/'

elif metric == 'attribute_attack':
    base_script= 'python  test.py --test_metric attribute_attack --mia_logit 1 --batch_size 64 ' + ' --dataset ' + str(dataset) + ' --attribute_domain ' + data_case

    res_dir= 'results/'+str(dataset)+'/attribute_attack_' + data_case + '/'  

elif metric  == 'acc':
    base_script= 'python test.py --test_metric acc ' + ' --dataset ' + str(dataset) + ' --acc_data_case ' + data_case
    res_dir= 'results/' + str(dataset) + '/acc_' + str(data_case) + '/'

elif metric  == 'match_score':
    base_script= 'python test.py --test_metric match_score ' + ' --dataset ' + str(dataset) + ' --match_func_data_case ' + data_case
    res_dir= 'results/' + str(dataset) + '/match_score_' + data_case + '/'

elif metric  == 'feat_eval':
    base_script= 'python test.py --test_metric feat_eval ' + ' --dataset ' + str(dataset) + ' --match_func_data_case ' + data_case
    res_dir= 'results/' + str(dataset) + '/feat_eval_' + data_case + '/'

elif metric  == 'feat_eval_rand':
    base_script= 'python test.py --test_metric feat_eval ' + ' --dataset ' + str(dataset) + ' --match_func_data_case ' + data_case + ' --match_case 0.0 '
    res_dir= 'results/' + str(dataset) + '/feat_eval_rand_' + data_case + '/'
    
    
#Train Domains 30, 45 case
if train_case == 'train_abl_2':
    base_script+= ' --train_domains 30 45'
    res_dir= res_dir[:-1] +'_30_45/'

#Train Domains 30, 45, 60 case
if train_case == 'train_abl_3':
    base_script+= ' --train_domains 30 45 60'
    res_dir= res_dir[:-1] +'_30_45_60/'            

#Test on 30, 45 angles instead of the standard 0, 90
if test_case  == 'test_common':
    base_script += ' --test_domains 30 45'
    res_dir+= 'test_common_domains/'

    
#Differential Privacy
if args.dp_noise:
    base_script += ' --dp_noise ' + str(args.dp_noise) + ' --dp_epsilon ' + str(args.dp_epsilon) + ' --dp_attach_opt ' + str(args.dp_attach_opt) + ' ' 
    res_dir= res_dir[:-1] + '_epsilon_' + str(args.dp_epsilon) + '/'
    
if not os.path.exists(res_dir):
    os.makedirs(res_dir)        

#Dataset Specific Modification
if dataset == 'rot_mnist_spur':
    base_script+= ' --img_c 3 '        

for method in methods:
    case= res_dir + str(method)
    
    print('Metric', metric, 'Method: ', method, ' Train Domains: ', train_case)
 
    if method == 'erm':
        script= base_script + ' --method_name erm_match --penalty_ws 0.0 --match_case 0.0 --epochs 25 ' +  ' > ' + case + '.txt'           
        os.system(script)

    elif method == 'rand':
        if dataset == 'rot_mnist_spur':        
            script= base_script + ' --method_name erm_match --penalty_ws 10.0 --match_case 0.0 --epochs 25 '  +  ' > ' + case + '.txt'   
        else:
            script= base_script + ' --method_name erm_match --penalty_ws 0.1 --match_case 0.0 --epochs 25 ' +  ' > ' + case + '.txt'       
        os.system(script)

    elif method == 'approx_25':
        script= base_script + ' --method_name erm_match --penalty_ws 0.1 --match_case 0.25 --epochs 25 ' +  ' > ' + case + '.txt'
        os.system(script)

    elif method == 'approx_50':
        script= base_script + ' --method_name erm_match --penalty_ws 0.1 --match_case 0.50 --epochs 25 ' +  ' > ' + case + '.txt'
        os.system(script)

    elif method == 'approx_75':
        script= base_script + ' --method_name erm_match --penalty_ws 0.1 --match_case 0.75 --epochs 25 ' + ' > ' + case + '.txt'
        os.system(script)

    elif method == 'perf':
        if dataset == 'rot_mnist_spur':                   
            script= base_script + ' --method_name erm_match --penalty_ws 10.0 --match_case 1.0 --epochs 25 ' +  ' > ' + case + '.txt' 
        else:
            script= base_script + ' --method_name erm_match --penalty_ws 0.1 --match_case 1.0 --epochs 25 ' +  ' > ' + case + '.txt' 
        os.system(script)

    elif method == 'matchdg':
        if metric == 'train':
            script= base_script + ' --method_name matchdg_ctr --match_case 0.0 --match_flag 1 --epochs 50 --batch_size 64 --pos_metric cos  --match_func_aug_case 1 ' +  ' > ' + res_dir + 'matchdg_ctr' + '.txt'  
            os.system(script)
        
        script= base_script + ' --method_name matchdg_erm --penalty_ws 0.1 --match_case -1 --ctr_match_case 0.0 --ctr_match_flag 1 --ctr_match_interrupt 5 --ctr_model_name resnet18 --epochs 25 ' +  ' > ' + case + '.txt'           
        os.system(script)

    elif method == 'csd':
        script= base_script + ' --method_name csd --penalty_ws 0.0 --match_case 0.0 --rep_dim 512 --epochs 25 ' +  ' > ' + case + '.txt'
        os.system(script)

    elif method == 'irm':
        script= base_script + ' --method_name irm --match_case 0.0 --penalty_irm 1.0 --penalty_s 5 --epochs 25 ' + ' > ' + case + '.txt'
        os.system(script)        

