import os
import sys
import argparse

# Input Parsing
parser = argparse.ArgumentParser()
parser.add_argument('--case', type=str, default='train', 
                    help='Case: train; test')
parser.add_argument('--slab_noise', type=float, default=0.0, 
                    help='Probability of corrupting slab features')
parser.add_argument('--methods', nargs='+', type=str, default=['erm', 'irm', 'csd', 'rand', 'matchdg', 'perf', 'mask_linear'],
                    help='List of methods')

args = parser.parse_args()

case= args.case
slab_noise= args.slab_noise
methods= args.methods

metrics= ['train-auc', 'auc', 'loss']
total_seed= 3

if case == 'train':
    
    base_script= 'python train.py --dataset slab --model_name slab --batch_size 128 --lr 0.1 --epochs 100 --out_classes 2 --train_domains 0.0 0.10 --test_domains 0.90 --slab_data_dim 2 '
    
    base_script= base_script + ' --slab_noise ' + str(slab_noise) +  ' --n_runs ' + str(total_seed)    

    for method in methods:

        if method == 'erm':
            script= base_script + ' --method_name erm_match --match_case 0.0 --penalty_ws 0.0 '
        elif method == 'irm':
            script= base_script + ' --method_name irm --match_case 0.0 --penalty_irm 10.0 --penalty_s 2 '
        elif method == 'csd':
            script= base_script + ' --method_name csd --match_case 0.0 --penalty_ws 0.0 --rep_dim 100 '
        elif method == 'rand':
            script= base_script + ' --method_name erm_match --match_case 0.0 --penalty_ws 1.0  '
        elif method == 'perf':
            script= base_script + ' --method_name erm_match --match_case 1.0 --penalty_ws 1.0 '        
        elif method == 'mask_linear':
            script= base_script + ' --method_name mask_linear --match_case 0.0 --penalty_ws 0.0 '
        elif method == 'matchdg':
            #CTR Phase
            script = base_script + ' --method_name matchdg_ctr --batch_size 512 --match_case 0.0 --match_flag 1 --match_interrupt 5 --pos_metric cos '
            os.system(script)
            
            #ERM Phase
            script = base_script + ' --method_name matchdg_erm --match_case -1 --penalty_ws 1.0 --ctr_match_case 0.0 --ctr_match_flag 1 --ctr_match_interrupt 5 --ctr_model_name slab '            
            
        os.system(script)
        
elif case == 'test':    

    for metric in metrics:

        if metric == 'auc':
            base_script= 'python test_slab.py --train_domains 0.0 0.10 '

        if metric == 'train-auc':
            base_script= 'python test_slab.py --train_domains 0.0 0.10 --acc_data_case train '            
        elif metric == 'mi':
            base_script= 'python  test.py --test_metric mia --mia_logit 1 --mia_sample_size 400 --dataset slab --model_name slab --out_classes 2 --train_domains 0.0 0.10 '
        elif metric == 'entropy':
            base_script= 'python  test.py --test_metric privacy_entropy --mia_sample_size 400 --dataset slab --model_name slab --out_classes 2 --train_domains 0.0 0.10 '
        elif metric == 'loss':
            base_script= 'python  test.py --test_metric privacy_loss_attack --mia_sample_size 400 --dataset slab --model_name slab --out_classes 2 --train_domains 0.0 0.10 '
        elif metric == 'attribute':
            base_script= 'python  test.py --test_metric attribute_attack --mia_logit 1 --attribute_domain 0 --dataset slab --model_name slab --out_classes 2 --train_domains 0.0 0.10 '           

        base_script= base_script + ' --slab_noise ' + str(slab_noise) + ' --n_runs ' + str(total_seed)
        res_dir= 'results/slab/slab_noise_' + str(slab_noise) + '/'

        if not os.path.exists(res_dir):
            os.makedirs(res_dir)
            
        for method in methods:

            if method == 'erm':
                upd_script= base_script + ' --method_name erm_match --match_case 0.0 --penalty_ws 0.0 '
            elif method == 'irm':
                upd_script= base_script + ' --method_name irm --match_case 0.0 --penalty_irm 10.0 --penalty_s 2 '
            elif method == 'csd':
                upd_script= base_script + ' --method_name csd --match_case 0.0 --penalty_ws 0.0 --rep_dim 100 '
            elif method == 'rand':
                upd_script= base_script + ' --method_name erm_match --match_case 0.0 --penalty_ws 1.0  '
            elif method == 'perf':
                upd_script= base_script + ' --method_name erm_match --match_case 1.0 --penalty_ws 1.0 '        
            elif method == 'mask_linear':
                upd_script= base_script + ' --method_name mask_linear --match_case 0.0 --penalty_ws 0.0 '
                
            elif method == 'matchdg':
                upd_script = base_script + ' --method_name matchdg_erm --match_case -1 --penalty_ws 1.0 --ctr_match_case 0.0 --ctr_match_flag 1 --ctr_match_interrupt 5 --ctr_model_name slab '                
                
            for test_domain in [0.2, 0.9]:
                script= upd_script + ' --test_domains ' + str(test_domain) + ' > ' + res_dir + str(method) + '-' + str(metric) + '-' + str(test_domain) + '.txt'
                os.system(script)
