# #MatchDG CTR Phase: RotMNIST
# python train.py --dataset rot_mnist --method_name matchdg_ctr --match_case 0.01 --match_flag 1 --epochs 30 --batch_size 64 --pos_metric cos

# python train.py --dataset rot_mnist --method_name matchdg_ctr --match_case 1.0 --match_flag 1 --epochs 30 --batch_size 64 --pos_metric cos

# python train.py --dataset rot_mnist --method_name matchdg_ctr --match_case 0.01 --match_flag 0 --epochs 30 --batch_size 64 --pos_metric cos

# #MatchDG CTR Phase: FashionMNIST
# python train.py --dataset fashion_mnist --method_name matchdg_ctr --match_case 0.01 --match_flag 1 --epochs 30 --batch_size 64 --pos_metric cos

# python train.py --dataset fashion_mnist --method_name matchdg_ctr --match_case 1.0 --match_flag 1 --epochs 30 --batch_size 64 --pos_metric cos

# python train.py --dataset fashion_mnist --method_name matchdg_ctr --match_case 0.01 --match_flag 0 --epochs 30 --batch_size 64 --pos_metric cos



python train.py --dataset rot_mnist --method_name matchdg_ctr --match_case 0.01 --match_flag 1 --epochs 100 --batch_size 128 --pos_metric cos

python train.py --dataset rot_mnist --method_name matchdg_erm --match_case -1 --penalty_ws 0.1 --epochs 25 --ctr_match_case 0.01 --ctr_match_flag 1 --ctr_match_interrupt 5 --ctr_model_name resnet18
