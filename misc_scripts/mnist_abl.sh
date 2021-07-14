##Rotated MNIST

#Perfect MatchDG
python train.py --dataset rot_mnist --method_name matchdg_ctr --match_case 1.0 --match_flag 1 --epochs 50 --batch_size 64 --pos_metric cos

#Non Interative MatchDG
python train.py --dataset rot_mnist --method_name matchdg_ctr --match_case 0.0 --match_flag 0 --epochs 50 --batch_size 64 --pos_metric cos


##Fashion MNIST

#Perfect MatchDG
python train.py --dataset fashion_mnist --method_name matchdg_ctr --match_case 1.0 --match_flag 1 --epochs 50 --batch_size 64 --pos_metric cos

#Non Interative MatchDG
python train.py --dataset fashion_mnist --method_name matchdg_ctr --match_case 0.0 --match_flag 0 --epochs 50 --batch_size 64 --pos_metric cos
