# Chex Evaluation
python train.py --dataset chestxray --method_name erm_match --match_case 0.01 --train_domains nih_trans kaggle_trans  --test_domains chex  --out_classes 2 --perfect_match 0 --img_c 3 --pre_trained 1 --epochs 40  --lr 0.001 --batch_size 16 --penalty_ws 0.0 --model_name densenet121 --n_runs 2 > erm_chex.txt

python train.py --dataset chestxray --method_name erm_match --match_case 0.01 --train_domains nih_trans kaggle_trans  --test_domains chex  --out_classes 2 --perfect_match 0 --img_c 3 --pre_trained 1 --epochs 40  --lr 0.001 --batch_size 16 --penalty_ws 50.0 --model_name densenet121 --n_runs 2 > rand_chex.txt

python train.py --dataset chestxray --method_name csd --match_case 0.01 --train_domains nih_trans kaggle_trans  --test_domains chex  --out_classes 2 --perfect_match 0 --img_c 3 --pre_trained 1 --epochs 40  --lr 0.001 --batch_size 16 --penalty_ws 0.0 --model_name densenet121 --n_runs 2 --rep_dim 1024 > csd_chex.txt

python train.py --dataset chestxray --method_name irm --match_case 0.01 --train_domains nih_trans kaggle_trans  --test_domains chex  --out_classes 2 --perfect_match 0 --img_c 3 --pre_trained 1 --epochs 40  --lr 0.001 --batch_size 16 --penalty_ws 50.0 --penalty_s 5 --model_name densenet121 --n_runs 2 > irm_chex.txt


# NIH Evaluation
python train.py --dataset chestxray --method_name erm_match --match_case 0.01 --train_domains chex_trans kaggle_trans  --test_domains nih  --out_classes 2 --perfect_match 0 --img_c 3 --pre_trained 1 --epochs 40  --lr 0.001 --batch_size 16 --penalty_ws 0.0 --model_name densenet121 --n_runs 2 > erm_nih.txt

python train.py --dataset chestxray --method_name erm_match --match_case 0.01 --train_domains chex_trans kaggle_trans  --test_domains nih  --out_classes 2 --perfect_match 0 --img_c 3 --pre_trained 1 --epochs 40  --lr 0.001 --batch_size 16 --penalty_ws 50.0 --model_name densenet121 --n_runs 2 > rand_nih.txt

python train.py --dataset chestxray --method_name csd --match_case 0.01 --train_domains chex_trans kaggle_trans  --test_domains nih  --out_classes 2 --perfect_match 0 --img_c 3 --pre_trained 1 --epochs 40  --lr 0.001 --batch_size 16 --penalty_ws 0.0 --model_name densenet121 --n_runs 2 --rep_dim 1024 > csd_nih.txt

python train.py --dataset chestxray --method_name irm --match_case 0.01 --train_domains chex_trans kaggle_trans  --test_domains nih  --out_classes 2 --perfect_match 0 --img_c 3 --pre_trained 1 --epochs 40  --lr 0.001 --batch_size 16 --penalty_ws 50.0 --penalty_s 5 --model_name densenet121 --n_runs 2 > irm_nih.txt


# Kaggle Evaluation
python train.py --dataset chestxray --method_name erm_match --match_case 0.01 --train_domains nih_trans chex_trans  --test_domains kaggle  --out_classes 2 --perfect_match 0 --img_c 3 --pre_trained 1 --epochs 40  --lr 0.001 --batch_size 16 --penalty_ws 0.0 --model_name densenet121 --n_runs 2 > erm_kaggle.txt

python train.py --dataset chestxray --method_name erm_match --match_case 0.01 --train_domains nih_trans chex_trans  --test_domains kaggle  --out_classes 2 --perfect_match 0 --img_c 3 --pre_trained 1 --epochs 40  --lr 0.001 --batch_size 16 --penalty_ws 50.0 --model_name densenet121 --n_runs 2 > rand_kaggle.txt

python train.py --dataset chestxray --method_name csd --match_case 0.01 --train_domains nih_trans chex_trans  --test_domains kaggle  --out_classes 2 --perfect_match 0 --img_c 3 --pre_trained 1 --epochs 40  --lr 0.001 --batch_size 16 --penalty_ws 0.0 --model_name densenet121 --n_runs 2 --rep_dim 1024 > csd_kaggle.txt

python train.py --dataset chestxray --method_name irm --match_case 0.01 --train_domains nih_trans chex_trans  --test_domains kaggle  --out_classes 2 --perfect_match 0 --img_c 3 --pre_trained 1 --epochs 40  --lr 0.001 --batch_size 16 --penalty_ws 50.0 --penalty_s 5 --model_name densenet121 --n_runs 2 > irm_kaggle.txt


# python train.py --dataset chestxray --method_name hybrid --match_case 0.01 --train_domains nih_trans chex_trans  --test_domains kaggle_trans  --out_classes 2 --perfect_match 0 --img_c 3 --pre_trained 1 --epochs 40  --lr 0.001 --batch_size 16 --penalty_ws 0.0 --model_name densenet121 --n_runs 3 --penalty_aug 10.0 > perf_chex_trans.txt






# python train.py --dataset chestxray --method_name erm_match --match_case 0.01 --train_domains nih_trans chex_trans  --test_domains kaggle  --out_classes 2 --perfect_match 0 --img_c 3 --pre_trained 1 --epochs 40  --lr 0.001 --batch_size 32 --penalty_ws 0.0 --model_name densenet121 --n_runs 1 > erm_chex.txt

# python train.py --dataset chestxray --method_name erm_match --match_case 0.01 --train_domains nih_trans chex_trans  --test_domains kaggle  --out_classes 2 --perfect_match 0 --img_c 3 --pre_trained 1 --epochs 40  --lr 0.001 --batch_size 32 --penalty_ws 1.0 --model_name densenet121 --n_runs 1 > rand_chex.txt

# python train.py --dataset chestxray --method_name csd --match_case 0.01 --train_domains nih_trans chex_trans  --test_domains kaggle  --out_classes 2 --perfect_match 0 --img_c 3 --pre_trained 1 --epochs 40  --lr 0.001 --batch_size 32 --penalty_ws 0.0 --model_name densenet121 --n_runs 1 --rep_dim 1024 > csd_chex.txt

# python train.py --dataset chestxray --method_name irm --match_case 0.01 --train_domains nih_trans chex_trans  --test_domains kaggle  --out_classes 2 --perfect_match 0 --img_c 3 --pre_trained 1 --epochs 40  --lr 0.001 --batch_size 32 --penalty_ws 1.0 --penalty_s 5 --model_name densenet121 --n_runs 1 > irm_chex.txt

# python train.py --dataset chestxray --method_name hybrid --match_case 0.01 --train_domains nih_trans chex_trans  --test_domains kaggle  --out_classes 2 --perfect_match 0 --img_c 3 --pre_trained 1 --epochs 40  --lr 0.001 --batch_size 16 --penalty_ws 0.0 --model_name densenet121 --n_runs 1 --penalty_aug 10.0 > perf_chex.txt

# #Kaggle Evaluation
# python train.py --dataset chestxray --method_name erm_match --match_case 0.01 --train_domains nih chex  --test_domains kaggle  --out_classes 2 --perfect_match 0 --img_c 3 --pre_trained 1 --epochs 40  --lr 0.001 --penalty_ws 0.0 --model_name densenet121 > erm_.txt

# python train.py --dataset chestxray --method_name erm_match --match_case 0.01 --train_domains nih chex  --test_domains kaggle  --out_classes 2 --perfect_match 0 --img_c 3 --pre_trained 1 --epochs 40  --lr 0.001 --penalty_ws 1.0 --model_name densenet121 > rand_chex.txt

# python train.py --dataset chestxray --method_name csd --match_case 0.01 --train_domains nih chex  --test_domains kaggle  --out_classes 2 --perfect_match 0 --img_c 3 --pre_trained 1 --epochs 40  --lr 0.001 --penalty_ws 0.0 --model_name densenet121 > csd_chex.txt

# python train.py --dataset chestxray --method_name irm --match_case 0.01 --train_domains nih chex  --test_domains kaggle  --out_classes 2 --perfect_match 0 --img_c 3 --pre_trained 1 --epochs 40  --lr 0.001 --penalty_ws 1.0 --penalty_s 5 --model_name densenet121 > irm_chex.txt


# #NIH Evaluation
# python train.py --dataset chestxray --method_name erm_match --match_case 0.01 --train_domains kaggle chex  --test_domains nih  --out_classes 2 --perfect_match 0 --img_c 3 --pre_trained 1 --epochs 40  --lr 0.001 --penalty_ws 0.0 --model_name densenet121 > erm_chex.txt

# python train.py --dataset chestxray --method_name erm_match --match_case 0.01 --train_domains kaggle chex  --test_domains nih  --out_classes 2 --perfect_match 0 --img_c 3 --pre_trained 1 --epochs 40  --lr 0.001 --penalty_ws 1.0 --model_name densenet121 > rand_chex.txt

# python train.py --dataset chestxray --method_name csd --match_case 0.01 --train_domains kaggle chex  --test_domains nih  --out_classes 2 --perfect_match 0 --img_c 3 --pre_trained 1 --epochs 40  --lr 0.001 --penalty_ws 0.0 --model_name densenet121 > csd_chex.txt

# python train.py --dataset chestxray --method_name irm --match_case 0.01 --train_domains kaggle chex  --test_domains nih  --out_classes 2 --perfect_match 0 --img_c 3 --pre_trained 1 --epochs 40  --lr 0.001 --penalty_ws 1.0 --penalty_s 5 --model_name densenet121 > irm_chex.txt

#Alone
# python train.py --dataset chestxray --method_name erm_match --match_case 0.01 --train_domains nih --test_domains nih --out_classes 2 --perfect_match 0 --img_c 3 --pre_trained 1 --epochs 40  --lr 0.001 --batch_size 32 --penalty_ws 0.0 --model_name densenet121 --n_runs 1 > nih_alone.txt

# python train.py --dataset chestxray --method_name erm_match --match_case 0.01 --train_domains chex --test_domains chex --out_classes 2 --perfect_match 0 --img_c 3 --pre_trained 1 --epochs 40  --lr 0.001 --batch_size 32 --penalty_ws 0.0 --model_name densenet121 --n_runs 1 > chex_alone.txt

# python train.py --dataset chestxray --method_name erm_match --match_case 0.01 --train_domains kaggle --test_domains kaggle --out_classes 2 --perfect_match 0 --img_c 3 --pre_trained 1 --epochs 40  --lr 0.001 --batch_size 32 --penalty_ws 0.0 --model_name densenet121 --n_runs 1 > kaggle_alone.txt