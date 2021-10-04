import os
import sys

# Input Parsing
parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, default='hybrid', 
                    help='erm; rand; matchdg_ctr; matchdg_erm; hybrid')
parser.add_argument('--model', type=str, default='resnet18', 
                    help='alexnet; resnet18; resnet50')
args = parser.parse_args()

method= args.method
model= args.model
domains= ['photo', 'art_painting', 'cartoon', 'sketch']

if method == 'erm' or method == 'rand':
    base_script= 'python train.py --dataset pacs  --method_name erm_match --match_case 0.0 --test_metric acc --out_classes 7 --perfect_match 0 --img_c 3 --pre_trained 1 --epochs 50 --weight_decay 0.001 '

elif method == 'matchdg_ctr':
    base_script= 'python train.py --dataset pacs --method_name matchdg_ctr --match_case 0.0 --match_flag 1 --pos_metric cos --out_classes 7 --perfect_match 0 --img_c 3 --pre_trained 1 --epochs 50 --batch_size 32  --match_func_aug_case 1  '

elif method == 'matchdg_erm':
    base_script= 'python train.py --dataset pacs --method_name matchdg_erm --match_case -1 --ctr_match_case 0.0 --ctr_match_flag 1 --ctr_match_interrupt 5 --ctr_model_name resnet50 --out_classes 7 --perfect_match 0 --img_c 3 --pre_trained 1 --epochs 50 --weight_decay 0.001 '

elif method == 'hybrid':
    base_script= 'python train.py --dataset pacs --method_name hybrid --match_case -1 --ctr_match_case 0.0 --ctr_match_flag 1 --ctr_match_interrupt 5 --ctr_model_name resnet50 --out_classes 7 --perfect_match 0 --img_c 3 --pre_trained 1 --epochs 50 --weight_decay 0.001 '
    
    
for test_domain in domains:
    
    res_dir= 'results/pacs/logs/'+ test_domain + '/'
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    
    train_domains=''
    for d in domains:
        if d != test_domain:
            train_domains+= str(d) + ' '
    print(train_domains)    
    
    if method == 'erm':        
        if model == 'resnet18':
            if test_domain == 'photo':
                lr= 0.001 
                penalty= 0.0
            elif test_domain == 'art_painting':
                lr= 0.01 
                penalty= 0.0
            elif test_domain == 'cartoon':
                lr= 0.01 
                penalty= 0.0
            elif test_domain == 'sketch':
                lr= 0.01 
                penalty= 0.0

        elif model == 'resnet50':
            if test_domain == 'photo':
                lr= 0.0005 
                penalty= 0.0
            elif test_domain == 'art_painting':
                lr= 0.01 
                penalty= 0.0
            elif test_domain == 'cartoon':
                lr= 0.01
                penalty= 0.0
            elif test_domain == 'sketch':
                lr= 0.01 
                penalty= 0.0

        elif model == 'alexnet':
            if test_domain == 'photo':
                lr= 0.0005 
                penalty= 0.0
            elif test_domain == 'art_painting':
                lr= 0.001 
                penalty= 0.0
            elif test_domain == 'cartoon':
                lr= 0.001
                penalty= 0.0
            elif test_domain == 'sketch':
                lr= 0.0005
                penalty= 0.0
                
    elif method == 'rand':        
        if model == 'resnet18':
            if test_domain == 'photo':
                lr= 0.001 
                penalty= 5.0
            elif test_domain == 'art_painting':
                lr= 0.01 
                penalty= 0.1
            elif test_domain == 'cartoon':
                lr= 0.001 
                penalty= 5.0
            elif test_domain == 'sketch':
                lr= 0.01 
                penalty= 0.5

        elif model == 'resnet50':
            if test_domain == 'photo':
                lr= 0.0005 
                penalty= 5.0
            elif test_domain == 'art_painting':
                lr= 0.01 
                penalty= 0.1
            elif test_domain == 'cartoon':
                lr= 0.01
                penalty= 0.01
            elif test_domain == 'sketch':
                lr= 0.01 
                penalty= 0.1            
                
        elif model == 'alexnet':
            if test_domain == 'photo':
                lr= 0.0005 
                penalty= 0.1
            elif test_domain == 'art_painting':
                lr= 0.001 
                penalty= 0.1
            elif test_domain == 'cartoon':
                lr= 0.001 
                penalty= 0.5
            elif test_domain == 'sketch':
                lr= 0.001 
                penalty= 0.5

    elif method == 'matchdg_erm':        
        if model == 'resnet18':
            if test_domain == 'photo':
                lr= 0.0005 
                penalty= 1.0
            elif test_domain == 'art_painting':
                lr= 0.001 
                penalty= 5.0
            elif test_domain == 'cartoon':
                lr= 0.001 
                penalty= 1.0
            elif test_domain == 'sketch':
                lr= 0.01 
                penalty= 0.5

        elif model == 'resnet50':
            if test_domain == 'photo':
                lr= 0.0005 
                penalty= 0.01
            elif test_domain == 'art_painting':
                lr= 0.001 
                penalty= 0.1
            elif test_domain == 'cartoon':
                lr= 0.001
                penalty= 0.01
            elif test_domain == 'sketch':
                lr= 0.0005
                penalty= 5.0    
                
        elif model == 'alexnet':
            if test_domain == 'photo':
                lr= 0.0005 
                penalty= 0.1
            elif test_domain == 'art_painting':
                lr= 0.001 
                penalty= 1.0
            elif test_domain == 'cartoon':
                lr= 0.001
                penalty= 1.0
            elif test_domain == 'sketch':
                lr= 0.001 
                penalty= 0.1

                
    elif method == 'hybrid':        
        if model == 'resnet18':
            if test_domain == 'photo':
                lr= 0.0005 
                penalty= 0.1
                penalty_aug= 0.1
            elif test_domain == 'art_painting':
                lr= 0.001 
                penalty= 0.01
                penalty_aug= 0.1
            elif test_domain == 'cartoon':
                lr= 0.001 
                penalty= 0.1
                penalty_aug= 0.1
            elif test_domain == 'sketch':
                lr= 0.01 
                penalty= 0.01
                penalty_aug= 0.1

        elif model == 'resnet50':
            if test_domain == 'photo':
                lr= 0.0005 
                penalty= 0.1
                penalty_aug= 0.1
            elif test_domain == 'art_painting':
                lr= 0.001 
                penalty= 0.01
                penalty_aug= 0.1
            elif test_domain == 'cartoon':
                lr= 0.0005
                penalty= 0.01
                penalty_aug= 0.1
            elif test_domain == 'sketch':
                lr= 0.001 
                penalty= 0.01
                penalty_aug= 0.1
                
        elif model == 'alexnet':
            if test_domain == 'photo':
                lr= 0.0005 
                penalty= 0.1
                penalty_aug= 0.1
            elif test_domain == 'art_painting':
                lr= 0.001 
                penalty= 0.01
                penalty_aug= 0.1
            elif test_domain == 'cartoon':
                lr= 0.001
                penalty= 0.01
                penalty_aug= 0.1
            elif test_domain == 'sketch':
                lr= 0.001 
                penalty= 0.01
                penalty_aug= 0.1
                
                
    if method == 'matchdg_ctr':
        script= base_script + ' --train_domains ' + train_domains + ' --test_domains ' + test_domain + ' --model_name ' + str(model)
    else:
        script= base_script + ' --train_domains ' + train_domains + ' --test_domains ' + test_domain + ' --lr ' + str(lr) + ' --penalty_ws ' + str(penalty) + ' --model_name ' + str(model)    
    
    #TODO: Add penalty_aug for hybrid method
    if method == 'hybrid':
        script= script + ' --penalty_aug ' + str(penalty_aug)
    
    #TOOD: Figure out the appropriate batch size
    if model == 'alexnet':
        script= script + ' --img_w 256 ' + ' --img_h 256 '
    
    save_dir= res_dir + str(method) + '_' + str(model) + '.txt'
    script= script + ' > ' + save_dir    
    os.system(script)
