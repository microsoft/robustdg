python train.py --dataset rot_mnist --method_name erm_match --match_case 0.01 --penalty_ws 1.0 --epochs 50 --model_name domain_bed_mnist --img_h 28 --img_w 28 --train_domains 15 30 45 60 75 --test_domains 0 > rand_1.txt

python train.py --dataset rot_mnist --method_name erm_match --match_case 0.30 --penalty_ws 1.0 --epochs 50 --model_name domain_bed_mnist --img_h 28 --img_w 28 --train_domains 15 30 45 60 75 --test_domains 0 > mdg_1.txt

python train.py --dataset rot_mnist --method_name erm_match --match_case 1.0 --penalty_ws 1.0 --epochs 50 --model_name domain_bed_mnist --img_h 28 --img_w 28 --train_domains 15 30 45 60 75 --test_domains 0 > perf_1.txt

python train.py --dataset rot_mnist --method_name erm_match --match_case 0.01 --penalty_ws 0.1 --epochs 50 --model_name domain_bed_mnist --img_h 28 --img_w 28 --train_domains 15 30 45 60 75 --test_domains 0 > rand_0.1.txt

python train.py --dataset rot_mnist --method_name erm_match --match_case 0.30 --penalty_ws 0.1 --epochs 50 --model_name domain_bed_mnist --img_h 28 --img_w 28 --train_domains 15 30 45 60 75 --test_domains 0 > mdg_0.1.txt

python train.py --dataset rot_mnist --method_name erm_match --match_case 1.0 --penalty_ws 0.1 --epochs 50 --model_name domain_bed_mnist --img_h 28 --img_w 28 --train_domains 15 30 45 60 75 --test_domains 0 > perf_0.1.txt