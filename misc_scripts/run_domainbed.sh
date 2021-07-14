#ERM MatchDG
python train.py --dataset rot_mnist --method_name erm_match --match_case 0.5 --penalty_ws 0.25 --model domain_bed_mnist --img_w 28 --img_h 28 --epochs 25 --train_domains 15 30 45 60 75 --test_domain 0

python train.py --dataset rot_mnist --method_name erm_match --match_case 0.5 --penalty_ws 0.25 --model domain_bed_mnist --img_w 28 --img_h 28 --epochs 25 --train_domains 0 15 30 45 60 --test_domain 75

#ERM RandMatch
python train.py --dataset rot_mnist --method_name erm_match --match_case 0.01 --penalty_ws 0.25 --model domain_bed_mnist --img_w 28 --img_h 28 --epochs 25 --train_domains 15 30 45 60 75 --test_domain 0

python train.py --dataset rot_mnist --method_name erm_match --match_case 0.01 --penalty_ws 0.25 --model domain_bed_mnist --img_w 28 --img_h 28 --epochs 25 --train_domains 0 15 30 45 60 --test_domain 75

#ERM PerfMatch
python train.py --dataset rot_mnist --method_name erm_match --match_case 1.0 --penalty_ws 0.25 --model domain_bed_mnist --img_w 28 --img_h 28 --epochs 25 --train_domains 15 30 45 60 75 --test_domain 0

python train.py --dataset rot_mnist --method_name erm_match --match_case 1.0 --penalty_ws 0.25 --model domain_bed_mnist --img_w 28 --img_h 28 --epochs 25 --train_domains 0 15 30 45 60 --test_domain 75


