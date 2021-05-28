import os
import sys

# method: rand, matchdg, perf
method= sys.argv[1]
model= sys.argv[2]
domains= ['photo', 'art_painting', 'cartoon', 'sketch']

if method == 'rand':

for test_domain in domains:
    base_script= 'python train.py --dataset pacs  --method_name erm_match --match_case 0.01 --test_metric acc --out_classes 7 --perfect_match 0 --img_c 3 --pre_trained 1 --epochs 50 --weight_decay 0.001 '

for test_domain in domains:
	
    train_domains=''
    for d in domains:
        if d != test_domain:
            train_domains+= str(d) + ' '
    print(train_domains)    

    res_dir= 'results/pacs/logs/'
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    
    if method == 'rand':        
        if model == 'resnet18':
            if test_domain == 'photo':
                lr= 0.001 
                penalty= 0.1
            elif test_domain == 'art_painting':
                lr= 0.01 
                penalty= 0.5
            elif test_domain == 'cartoon':
                lr= 0.01 
                penalty= 0.1
            elif test_domain == 'sketch':
                lr= 0.01 
                penalty= 0.1

        elif model == 'resnet50':
            if test_domain == 'photo':
                lr= 0.0005 
            elif test_domain == 'art_painting':
                lr= 0.001 
                penalty= 0.5
            elif test_domain == 'cartoon':
                lr= 0.0005
                penalty= 0.5
            elif test_domain == 'sketch':
                lr= 0.01 
                penalty= 0.1            
                
    script= base_script + ' --train_domains ' + train_domains + ' --test_domains ' + test_domain + ' --lr ' + str(lr) + ' --penalty_ws ' + str(penalty) + ' --model_name ' + str(model)
    os.system(script)
