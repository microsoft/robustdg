#RotMNIST
python train.py --dataset rot_mnist --method_name erm_match --match_case 0.01 --penalty_ws 0.0 --epochs 25
python train.py --dataset rot_mnist --method_name erm_match --match_case 0.01 --penalty_ws 0.1 --epochs 25
python train.py --dataset rot_mnist --method_name erm_match --match_case 0.25 --penalty_ws 0.1 --epochs 25
python train.py --dataset rot_mnist --method_name erm_match --match_case 0.50 --penalty_ws 0.1 --epochs 25
python train.py --dataset rot_mnist --method_name erm_match --match_case 1.0 --penalty_ws 0.1 --epochs 25
python train.py --dataset rot_mnist --method_name erm_match --match_case 0.75 --penalty_ws 0.1 --epochs 25
python train.py --dataset rot_mnist --method_name erm_match --match_case 0.01 --penalty_ws 0.0 --epochs 25

python train.py --dataset rot_mnist --method_name irm --match_case 0.01 --penalty_irm 1.0 --penalty_s 5 -- epochs 25
python train.py --dataset rot_mnist --method_name csd --match_case 0.01 --penalty_ws 0.0 -- epochs 25

# python train.py --dataset rot_mnist --method_name erm_match --match_case 0.01 --penalty_ws 0.0 --epochs 25 --train_domains 30 45
# python train.py --dataset rot_mnist --method_name erm_match --match_case 0.01 --penalty_ws 0.1 --epochs 25 --train_domains 30 45
# python train.py --dataset rot_mnist --method_name erm_match --match_case 0.25 --penalty_ws 0.1 --epochs 25 --train_domains 30 45
# python train.py --dataset rot_mnist --method_name erm_match --match_case 0.50 --penalty_ws 0.1 --epochs 25 --train_domains 30 45
# python train.py --dataset rot_mnist --method_name erm_match --match_case 0.75 --penalty_ws 0.1 --epochs 25 --train_domains 30 45
# python train.py --dataset rot_mnist --method_name erm_match --match_case 1.0 --penalty_ws 0.1 --epochs 25 --train_domains 30 45
# python train.py --dataset rot_mnist --method_name matchdg_ctr --match_case 0.01 --match_flag 1 --epochs 30 --batch_size 64 --pos_metric cos  
# python train.py --dataset rot_mnist --method_name matchdg_erm --penalty_ws 0.1 --match_case -1 --ctr_match_case 0.01 --ctr_match_flag 1 --ctr_match_interrupt 5  --n_runs 3 --epochs 25
# python train.py --dataset rot_mnist --method_name csd --match_case 0.01 --penalty_ws 0.0 -- epochs 25 --train_domains 30 45
# python train.py --dataset rot_mnist --method_name irm --match_case 0.01 --penalty_irm 1.0 --penalty_s 5 -- epochs 25 --train_domains 30 45
# python train.py --dataset rot_mnist --method_name matchdg_ctr --match_case 0.01 --match_flag 1 --epochs 30 --batch_size 64 --pos_metric cos  
# python train.py --dataset rot_mnist --method_name matchdg_erm --penalty_ws 0.1 --match_case -1 --ctr_match_case 0.01 --ctr_match_flag 1 --ctr_match_interrupt 5  --n_runs 3 --epochs 25

#FashionMNIST
# python train.py --dataset fashion_mnist --method_name erm_match --match_case 0.01 --penalty_ws 0.0 --epochs 25
# python train.py --dataset fashion_mnist --method_name erm_match --match_case 0.01 --penalty_ws 0.1 --epochs 25
# python train.py --dataset fashion_mnist --method_name erm_match --match_case 0.25 --penalty_ws 0.1 --epochs 25
# python train.py --dataset fashion_mnist --method_name erm_match --match_case 0.50 --penalty_ws 0.1 --epochs 25
# python train.py --dataset fashion_mnist --method_name erm_match --match_case 0.75 --penalty_ws 0.1 --epochs 25
# python train.py --dataset fashion_mnist --method_name erm_match --match_case 1.0 --penalty_ws 0.1 --epochs 25
# python train.py --dataset fashion_mnist --method_name csd --match_case 0.01 --penalty_ws 0.0 -- epochs 25


#ChestXRay

# python train.py --dataset chestxray --method_name matchdg_ctr --match_case 0.01 --match_flag 1 --epochs 3 --batch_size 32 --pos_metric cos --train_domains nih_trans chex_trans --test_domains kaggle_trans --out_classes 2 --perfect_match 0 --img_c 3 --pre_trained 1 --model_name densenet121 --n_runs 2

