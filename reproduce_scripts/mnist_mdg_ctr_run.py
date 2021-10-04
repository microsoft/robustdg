import os
import argparse

# Input Parsing
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='rot_mnist', 
                    help='Datasets: rot_mnist; fashion_mnist')
parser.add_argument('--iterative', type=int, default=1, 
                    help='Iterative updates to positive matches')
parser.add_argument('--perf_init', type=int, default=0, 
                    help='Positive matches Initialization to perfect matches')

args = parser.parse_args()
dataset= args.dataset
iterative= args.iterative
perf_init= args.perf_init

base_script= 'python train.py --epochs 50 --batch_size 64 --dataset ' + str(dataset)
script= base_script + '  --method_name matchdg_ctr --pos_metric cos --match_case ' +  str(perf_init) + ' --match_flag ' + str(iterative)
os.system(script)


#2) Script for evaluating the match function metrics
