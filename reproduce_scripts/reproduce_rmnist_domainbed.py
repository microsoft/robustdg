import os
import argparse

# Input Parsing
parser = argparse.ArgumentParser()
parser.add_argument('--methods', nargs='+', type=str, default=['rand', 'matchdg_ctr', 'matchdg_erm'], 
                    help='List of methods')

args = parser.parse_args()
methods= args.methods                    
domains= [0, 15, 30, 45, 60, 75]

for method in methods:
    
    if method == 'rand':
        base_script= 'python train.py --dataset rot_mnist --mnist_case domain_bed --method_name erm_match --perfect_match 0 --match_case 0.0 --penalty_ws 1.0 --epochs 25 --batch_size 128 --model_name domain_bed_mnist --img_h 28 --img_w 28 --total_matches_per_point 1000 '

    elif method == 'matchdg_ctr':
        base_script= 'python train.py --dataset rot_mnist --mnist_case domain_bed --method_name matchdg_ctr --perfect_match 0 --match_case 0.0 --match_flag 1 --epochs 50 --batch_size 512 --pos_metric cos --model_name domain_bed_mnist --img_h 28 --img_w 28 --match_func_aug_case 1 '

    elif method == 'matchdg_erm':
        base_script= 'python train.py --dataset rot_mnist --mnist_case domain_bed --method_name matchdg_erm --perfect_match 0 --match_case -1 --penalty_ws 1.0 --epochs 25 --batch_size 128 --ctr_match_case 0.0 --ctr_match_flag 1 --ctr_match_interrupt 5 --ctr_model_name domain_bed_mnist  --model_name domain_bed_mnist --img_h 28 --img_w 28 --total_matches_per_point 1000 '

    for test_domain in domains:

        train_domains=''
        for d in domains:
            if d != test_domain:
                train_domains+= str(d) + ' '
        print(train_domains)

        res_dir= 'results/rmnist_domain_bed/'
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)

        script= base_script + ' --train_domains ' + str(train_domains) + ' --test_domains ' + str(test_domain) 
        script= script + ' > ' + res_dir + method + '_' + str(test_domain) + '.txt'

        print('Method: ', method, ' Test Domain: ', test_domain)
        os.system(script)