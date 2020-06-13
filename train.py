import os
import argparse

# Input Parsing
parser = argparse.ArgumentParser(description='Evaluation')
parser.add_argument('--dataset', type=str, default='rot_mnist')
parser.add_argument('--domain_abl', type=int, default=0, help='0: No Abl; x: Train with x domains only')
parser.add_argument('--penalty_ws', type=float, default=0.1)
parser.add_argument('--match_case', type=float, default=1, help='0.01: Random Match; 1: Perfect Match')
parser.add_argument('--match_dg', type=int, default=0, help='0: ERM, ERM+Match; 1: MatcDG Phase 1; 2: MatchDG Phase 2')

args = parser.parse_args()

# Get results for ERM, ERM_RandomMatch, ERM_PerfectMatch 
if args.match_dg == 0:
    base_script= "python3 main-train.py --lr 0.01 --epochs 15 --batch_size 16 --penalty_w 0.0 --penalty_s -1 --penalty_same_ctr 0.0 --penalty_diff_ctr 0.0 --penalty_erm 1.0 --same_margin 1.0 --diff_margin 100.0 --save_logs 0 --test_domain 1  --seed  -1 --match_flag 0 --match_interrupt 35 --pre_trained 1 --method_name phi_match --out_classes 10 --n_runs 3 --pos_metric l2 --model mnist --perfect_match 1 --erm_base 1 --ctr_phase 0 --erm_phase 0 --penalty_ws_erm 0.0"

    script= base_script + ' --match_case ' + str(args.match_case) + ' --dataset ' + str(args.dataset) + ' --domain_abl ' + str(args.domain_abl) + ' --penalty_ws ' + str(args.penalty_ws) 
    os.system(script)

#Train MatchDG
else:
    
    base_script= "python3 main-train.py --lr 0.01 --epochs 30 --batch_size 64 --penalty_w 0.0 --penalty_s -1 --penalty_ws 0.0 --penalty_same_ctr 0.0 --penalty_diff_ctr 1.0 --penalty_erm 1.0 --same_margin 1.0 --diff_margin 1.5 --save_logs 0 --test_domain 1  --seed  -1 --match_flag 1 --match_interrupt 5 --pre_trained 1 --method_name phi_match --pos_metric cos --out_classes 10 --n_runs 2 --model mnist --perfect_match 1 --erm_base 0 --ctr_phase 1 --erm_phase 0 --epochs_erm 25 --penalty_ws_erm 0.1 --match_case_erm -1 --opt sgd"

    script = base_script + ' --domain_abl ' + str(args.domain_abl) + ' --dataset ' + str(args.dataset) + ' --match_case ' + str(args.match_case)
    os.system(script)
    
    # Don't need Match DG Phase 2 for Perfect Match Seed
    if args.match_case == 0.01:
        
        base_script= "python3 main-train.py --lr 0.01 --epochs 30 --batch_size 16 --penalty_w 0.0 --penalty_s -1 --penalty_ws 0.0 --penalty_same_ctr 0.0 --penalty_diff_ctr 1.0 --penalty_erm 1.0 --same_margin 1.0 --diff_margin 1.5 --save_logs 0 --test_domain 1  --seed  -1 --match_case 0.01 --match_flag 1 --match_interrupt 5 --pre_trained 1 --method_name phi_match --pos_metric cos --out_classes 10 --n_runs 2 --model mnist --perfect_match 1 --erm_base 0 --ctr_phase 0 --erm_phase 1 --epochs_erm 15 --penalty_ws_erm 0.1 --match_case_erm -1 --opt sgd"

        script= base_script + ' --dataset ' + str(args.dataset) + ' --domain_abl ' + str(args.domain_abl)
        os.system(script)