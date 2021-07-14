# # python3 metric_eval.py fashion_mnist mia
# # python3 metric_eval.py rot_mnist mia

# # python train.py --dataset pacs --method_name hybrid --match_case -1 --ctr_match_case 0.01 --ctr_match_flag 1 --ctr_match_interrupt 5 --n_runs 3 --train_domains photo cartoon sketch --test_domains art_painting --out_classes 7 --perfect_match 0 --img_c 3 --pre_trained 1 --epochs 50 --lr 0.001 --batch_size 16 --weight_decay 0.001 --penalty_ws 0.1 --penalty_aug 0.1 --model_name resnet18 --ctr_model_name resnet18

# #python train.py --dataset pacs --method_name hybrid --match_case -1 --ctr_match_case 0.01 --ctr_match_flag 1 --ctr_match_interrupt 5 --n_runs 3 --train_domains photo art_painting cartoon --test_domains sketch --out_classes 7 --perfect_match 0 --img_c 3 --pre_trained 1 --epochs 50 --lr 0.01 --batch_size 16 --weight_decay 0.001 --penalty_ws 0.1 --penalty_aug 0.1 --model_name resnet18 --ctr_model_name resnet18

# #python train.py --dataset pacs --method_name hybrid --match_case -1 --ctr_match_case 0.01 --ctr_match_flag 1 --ctr_match_interrupt 5 --n_runs 3 --train_domains photo cartoon sketch --test_domains art_painting --out_classes 7 --perfect_match 0 --img_c 3 --pre_trained 1 --epochs 50 --lr 0.001 --batch_size 16 --weight_decay 0.001 --penalty_ws 0.1 --penalty_aug 0.1 --model_name resnet18 --ctr_model_name resnet18


# #python3 metric_eval.py fashion_mnist acc test
# #python3 metric_eval.py fashion_mnist match_score test
# #python3 metric_eval.py fashion_mnist mia
# python3 metric_eval.py fashion_mnist privacy_entropy
# #python3 metric_eval.py fashion_mnist attribute_attack
# #python3 metric_eval.py fashion_mnist privacy_loss_attack


# python train.py --dataset rot_mnist --method_name erm_match --match_case 0.0 --penalty_ws 0.0 --epochs 25
# python train.py --dataset rot_mnist --method_name erm_match --match_case 0.0 --penalty_ws 0.1 --epochs 25
# python train.py --dataset rot_mnist --method_name erm_match --match_case 0.25 --penalty_ws 0.1 --epochs 25
# python train.py --dataset rot_mnist --method_name erm_match --match_case 0.50 --penalty_ws 0.1 --epochs 25
# python train.py --dataset rot_mnist --method_name erm_match --match_case 0.75 --penalty_ws 0.1 --epochs 25
# python train.py --dataset rot_mnist --method_name erm_match --match_case 1.0 --penalty_ws 0.1 --epochs 25

# python train.py --dataset fashion_mnist --method_name erm_match --match_case 0.0 --penalty_ws 0.0 --epochs 25
# python train.py --dataset fashion_mnist --method_name erm_match --match_case 0.0 --penalty_ws 0.1 --epochs 25
# python train.py --dataset fashion_mnist --method_name erm_match --match_case 0.25 --penalty_ws 0.1 --epochs 25
# python train.py --dataset fashion_mnist --method_name erm_match --match_case 0.50 --penalty_ws 0.1 --epochs 25
# python train.py --dataset fashion_mnist --method_name erm_match --match_case 0.75 --penalty_ws 0.1 --epochs 25
# python train.py --dataset fashion_mnist --method_name erm_match --match_case 1.0 --penalty_ws 0.1 --epochs 25

#python train.py --dataset rot_mnist --method_name irm --match_case 0.0 --penalty_irm 1.0 --penalty_s 5 --epochs 25

#python train.py --dataset rot_mnist --method_name csd --match_case 0.0 --penalty_ws 0.0 --epochs 25 --rep_dim 512

python train.py --dataset rot_mnist --method_name matchdg_ctr --match_case 0.0 --match_flag 1 --epochs 50 --batch_size 64 --pos_metric cos

python train.py --dataset rot_mnist --method_name matchdg_erm --match_case -1 --penalty_ws 0.1 --epochs 25 --ctr_match_case 0.0 --ctr_match_flag 1 --ctr_match_interrupt 5 --ctr_model_name resnet18
