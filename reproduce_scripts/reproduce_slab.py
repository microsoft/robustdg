import os
import sys

'''
argv1: Allowed Values (train, evaluate)
'''

case= sys.argv[1]
methods=['erm', 'mmd', 'coral', 'dann', 'c-mmd', 'c-coral', 'c-dann', 'rand', 'perf']
total_seed= 10

if case == 'train':
    base_script= 'python train.py --dataset slab --model_name slab --batch_size 128 --lr 0.1 --epochs 100 --out_classes 2 --train_domains 0.0 0.10 --test_domains 1.0 --slab_data_dim 2 --slab_noise 0.1 ' + ' --n_runs ' + str(total_seed)

elif case == 'evaluate':
    base_script= 'python test.py --test_metric per_domain_acc --acc_data_case train --dataset slab --model_name slab --batch_size 128 --lr 0.1 --epochs 100 --out_classes 2 --train_domains 0.0 0.10 --test_domains 1.0 --slab_data_dim 2 --slab_noise 0.1 ' + ' --n_runs ' + str(total_seed)

    
res_dir= 'results/slab/logs/'
if not os.path.exists(res_dir):
    os.makedirs(res_dir)


for method in methods:
    
    if method == 'mmd':
        script= base_script + ' --method_name mmd --gaussian 1 --conditional 0 --penalty_ws 0.1 '

    elif method == 'c-mmd':
        script= base_script + ' --method_name mmd --gaussian 1 --conditional 1 --penalty_ws 0.1 '

    elif method == 'coral':
        script= base_script + ' --method_name mmd --gaussian 0 --conditional 0  --penalty_ws 0.1 '

    elif method == 'c-coral':    
        script= base_script + ' --method_name mmd --gaussian 0 --conditional 1  --penalty_ws 0.1 '

    elif method == 'dann':
        script= base_script + ' --method_name dann --conditional 0 --penalty_ws 0.01 --grad_penalty 0.1 --d_steps_per_g_step 4 '

    elif method == 'c-dann':
        script= base_script + ' --method_name dann --conditional 1 --penalty_ws 0.01 --grad_penalty 1.0 --d_steps_per_g_step 2 '

    elif method == 'erm':
        script= base_script + ' --method_name erm_match --match_case 0.0 --penalty_ws 0.0 '

    elif method == 'rand':
        script= base_script + ' --method_name erm_match --match_case 0.0 --penalty_ws 1.0 '

    elif method == 'perf':
        script= base_script + ' --method_name erm_match --match_case 1.0 --penalty_ws 1.0 '

    script= script + ' > ' + res_dir + str(method) + '.txt'
    os.system(script)