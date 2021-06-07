import os
import sys

# method: rand, matchdg, perf
method= sys.argv[1]
domains= [0, 15, 30, 45, 60, 75]


if method == 'perf':
	base_script= 'python train.py --dataset rot_mnist --method_name erm_match --match_case 1.0 --penalty_ws 1.0 --epochs 100 --model_name lenet --img_h 32 --img_w 32 '

elif method == 'rand':
	base_script= 'python train.py --dataset rot_mnist --method_name erm_match --match_case 0.0 --penalty_ws 1.0 --epochs 100 --model_name lenet --img_h 32 --img_w 32 '

elif method == 'matchdg_ctr':
    base_script= 'python train.py --dataset rot_mnist --method_name matchdg_ctr --match_case 0.0 --match_flag 1 --epochs 100 --batch_size 64 --pos_metric cos --model_name resnet18'
    
elif method == 'matchdg_erm':
    base_script= 'python train.py --dataset rot_mnist --method_name matchdg_erm --match_case -1 --penalty_ws 1.0 --epochs 100 --ctr_match_case 0.0 --ctr_match_flag 1 --ctr_match_interrupt 5 --ctr_model_name resnet18  --model_name lenet --img_h 32 --img_w 32'

for test_domain in domains:
    train_domains=''
    for d in domains:
        if d != test_domain:
            train_domains+= str(d) + ' '
    print(train_domains)

    res_dir= 'results/rmnist_lenet/'
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
        
    script= base_script + ' --train_domains ' + str(train_domains) + ' --test_domains ' + str(test_domain) + ' > ' + res_dir + method + '_' + str(test_domain) + '.txt'
    
    print('Method: ', method, ' Test Domain: ', test_domain)
    os.system(script)