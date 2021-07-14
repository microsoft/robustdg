# python test.py --dataset rot_mnist --method_name erm_match --match_case 0.01 --penalty_ws 0.0 --test_metric adv_attack --penalty_diff_ctr 0 --adv_eps 5
# python test.py --dataset rot_mnist --method_name erm_match --match_case 0.01 --penalty_ws 0.1 --test_metric adv_attack --penalty_diff_ctr 0 --adv_eps 5
# python test.py --dataset rot_mnist --method_name erm_match --match_case 1.0 --penalty_ws 0.1 --test_metric adv_attack --penalty_diff_ctr 0 --adv_eps 5

# python test.py --dataset rot_mnist --method_name erm_match --match_case 0.01 --penalty_ws 0.0 --test_metric adv_attack --penalty_diff_ctr 0 --adv_eps 25
# python test.py --dataset rot_mnist --method_name erm_match --match_case 0.01 --penalty_ws 0.1 --test_metric adv_attack --penalty_diff_ctr 0 --adv_eps 25
# python test.py --dataset rot_mnist --method_name erm_match --match_case 1.0 --penalty_ws 0.1 --test_metric adv_attack --penalty_diff_ctr 0 --adv_eps 25

# python test.py --dataset rot_mnist --method_name erm_match --match_case 0.01 --penalty_ws 0.0 --test_metric adv_attack --penalty_diff_ctr 0 --adv_eps 50
# python test.py --dataset rot_mnist --method_name erm_match --match_case 0.01 --penalty_ws 0.1 --test_metric adv_attack --penalty_diff_ctr 0 --adv_eps 50
# python test.py --dataset rot_mnist --method_name erm_match --match_case 1.0 --penalty_ws 0.1 --test_metric adv_attack --penalty_diff_ctr 0 --adv_eps 50

# python test.py --dataset rot_mnist --method_name erm_match --match_case 0.01 --penalty_ws 0.0 --test_metric adv_attack --penalty_diff_ctr 0 --adv_eps 75
# python test.py --dataset rot_mnist --method_name erm_match --match_case 0.01 --penalty_ws 0.1 --test_metric adv_attack --penalty_diff_ctr 0 --adv_eps 75
# python test.py --dataset rot_mnist --method_name erm_match --match_case 1.0 --penalty_ws 0.1 --test_metric adv_attack --penalty_diff_ctr 0 --adv_eps 75

# python test.py --dataset rot_mnist --method_name erm_match --match_case 0.01 --penalty_ws 0.0 --test_metric adv_attack --penalty_diff_ctr 0 --adv_eps 100
# python test.py --dataset rot_mnist --method_name erm_match --match_case 0.01 --penalty_ws 0.1 --test_metric adv_attack --penalty_diff_ctr 0 --adv_eps 100
# python test.py --dataset rot_mnist --method_name erm_match --match_case 1.0 --penalty_ws 0.1 --test_metric adv_attack --penalty_diff_ctr 0 --adv_eps 100

python3 test.py --test_metric adv_attack --dataset fashion_mnist --method_name erm_match --match_case 1.0 --penalty_ws 0.1 --adv_eps 0.3 > perfect.txt

python3 test.py --test_metric adv_attack --dataset fashion_mnist --method_name erm_match --match_case 0.75 --penalty_ws 0.1 --adv_eps 0.3 > case_75.txt

python3 test.py --test_metric adv_attack --dataset fashion_mnist --method_name erm_match --match_case 0.5 --penalty_ws 0.1 --adv_eps 0.3 > case_50.txt

python3 test.py --test_metric adv_attack --dataset fashion_mnist --method_name erm_match --match_case 0.25 --penalty_ws 0.1 --adv_eps 0.3 > case_25.txt

python3 test.py --test_metric adv_attack --dataset fashion_mnist --method_name erm_match --match_case 0.01 --penalty_ws 0.1 --adv_eps 0.3 > random.txt

python3 test.py --test_metric adv_attack --dataset fashion_mnist --method_name erm_match --match_case 0.01 --penalty_ws 0.0 --adv_eps 0.3 > erm.txt

python  test.py --test_metric adv_attack --dataset fashion_mnist --method_name matchdg_erm --penalty_ws 0.1 --match_case -1 --ctr_match_case 0.01 --ctr_match_flag 1 --ctr_match_interrupt 5 --adv_eps 0.3 --n_runs 2 > match_dg.txt