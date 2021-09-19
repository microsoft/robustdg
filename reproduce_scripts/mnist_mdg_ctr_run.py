import os
#rot_mnist, fashion_mnist
dataset=sys.argv[1]

#TODO 
#1) Add another argparse arugment for deciding between perfect and non-iterative
#2) Script for evaluating the match function metrics

base_script= 'python train.py --epochs 50 --batch_size 64 --dataset ' + str(dataset)

#Perf MDG
script= base_script + '  --method_name matchdg_ctr --match_case 1.0 --match_flag 1  --pos_metric cos '
os.system(script)

#Non Iterative MDG
script= base_script + '  --method_name matchdg_ctr --match_case 0.0 --match_flag 0  --pos_metric cos '
os.system(script)