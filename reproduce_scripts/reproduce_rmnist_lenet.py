import os
import argparse

# Input Parsing
parser = argparse.ArgumentParser()
parser.add_argument('--methods', nargs='+', type=str, default=['erm', 'rand', 'matchdg_ctr', 'matchdg_erm', 'perf'], 
                    help='List of methods')

args = parser.parse_args()
methods= args.methods                    
domains= [0, 15, 30, 45, 60, 75]

for method in methods:
                    
    if method == 'perf':
        base_script= 'python train.py --dataset rot_mnist --mnist_case lenet --method_name erm_match --match_case 1.0 --penalty_ws 1.0 --epochs 100 --model_name lenet --img_h 32 --img_w 32 '

    elif method == 'erm':
        base_script= 'python train.py --dataset rot_mnist --mnist_case lenet --method_name erm_match --match_case 0.0 --penalty_ws 0.0 --epochs 100 --model_name lenet --img_h 32 --img_w 32 '

    elif method == 'rand':
        base_script= 'python train.py --dataset rot_mnist --mnist_case lenet --method_name erm_match --match_case 0.0 --penalty_ws 1.0 --epochs 100 --model_name lenet --img_h 32 --img_w 32 --total_matches_per_point 100 '

    elif method == 'matchdg_ctr':
        base_script= 'python train.py --dataset rot_mnist --mnist_case lenet --method_name matchdg_ctr --match_case 0.0 --match_flag 1 --epochs 50 --batch_size 512 --pos_metric cos --model_name lenet --img_h 32 --img_w 32 --match_func_aug_case 1 '

    elif method == 'matchdg_erm':
        base_script= 'python train.py --dataset rot_mnist --mnist_case lenet --method_name matchdg_erm --match_case -1 --penalty_ws 1.0 --epochs 100 --ctr_match_case 0.0 --ctr_match_flag 1 --ctr_match_interrupt 5 --ctr_model_name lenet  --model_name lenet --img_h 32 --img_w 32 --total_matches_per_point 100 '

    for test_domain in domains:

        train_domains=''
        for d in domains:
            if d != test_domain:
                train_domains+= str(d) + ' '
        print(train_domains)

        res_dir= 'results/rmnist_lenet/'
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)

        script= base_script + ' --train_domains ' + str(train_domains) + ' --test_domains ' + str(test_domain) 
        script= script + ' > ' + res_dir + method + '_' + str(test_domain) + '.txt'

        print('Method: ', method, ' Test Domain: ', test_domain)
        os.system(script)