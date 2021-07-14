# python train.py --dataset slab --model_name slab --method_name perf_match --batch_size 64 --lr 0.1 --penalty_ws 0.0 --epochs 30 --out_classes 2 --train_domains 0.0 0.10 --test_domains 0.50 --n_runs 3 > slab_temp/erm.txt

# python  test.py --test_metric acc --dataset slab --model_name slab --method_name perf_match --penalty_ws 0.0 --out_classes 2 --train_domains 0.0 0.10 --n_runs 3  --acc_data_case train

# python  test.py --test_metric slab_feat_eval --dataset slab --model_name slab --method_name perf_match --penalty_ws 0.0 --out_classes 2 --train_domains 0.0 0.10 --n_runs 3  --match_func_data_case train

# python test_slab.py --method_name perf_match --penalty_ws 0.0 --n_runs 3 --train_domains 0.0 0.10 --test_domain 0.3

# # python  test.py --test_metric mia --dataset slab --model_name slab --method_name perf_match --penalty_ws 0.0 --mia_logit 1 --mia_sample_size 400 --out_classes 2 --train_domains 0.0 0.10 --test_domains 0.50 --n_runs 3  > slab_temp/erm-mi.txt

# # python  test.py --test_metric privacy_entropy --dataset slab --model_name slab --method_name perf_match --penalty_ws 0.0 --mia_sample_size 400 --out_classes 2 --train_domains 0.0 0.10 --test_domains 0.50 --n_runs 3  > slab_temp/erm-entropy.txt

# # python  test.py --test_metric privacy_loss_attack --dataset slab --model_name slab --method_name perf_match --penalty_ws 0.0 --mia_sample_size 400 --out_classes 2 --train_domains 0.0 0.10 --test_domains 0.50 --n_runs 3  > slab_temp/erm-loss.txt



# python train.py --dataset slab --model_name slab --method_name rand_match --batch_size 64 --lr 0.1 --penalty_ws 1.0 --epochs 30 --out_classes 2 --train_domains 0.0 0.10 --test_domains 1.0 --n_runs 3  > slab_temp/rand.txt

# python  test.py --test_metric acc --dataset slab --model_name slab --method_name rand_match --penalty_ws 1.0 --out_classes 2 --train_domains 0.0 0.10 --n_runs 3  --acc_data_case train

# python test_slab.py --method_name rand_match --penalty_ws 1.0 --n_runs 3 --train_domains 0.0 0.10 --test_domain 0.3

# # python  test.py --test_metric mia --dataset slab --model_name slab --method_name rand_match --penalty_ws 10.0 --mia_logit 1 --mia_sample_size 400 --out_classes 2 --train_domains 0.0 0.10 --test_domains 0.50 --n_runs 3  > slab_temp/rand-mi.txt

# # python  test.py --test_metric privacy_entropy --dataset slab --model_name slab --method_name rand_match --penalty_ws 10.0 --mia_sample_size 400 --out_classes 2 --train_domains 0.0 0.10 --test_domains 0.50 --n_runs 3  > slab_temp/rand-entropy.txt

# # python  test.py --test_metric privacy_loss_attack --dataset slab --model_name slab --method_name rand_match --penalty_ws 10.0 --mia_sample_size 400 --out_classes 2 --train_domains 0.0 0.10 --test_domains 0.50 --n_runs 3  > slab_temp/rand-loss.txt



# python train.py --dataset slab --model_name slab --method_name perf_match --batch_size 64 --lr 0.1 --penalty_ws 1.0 --epochs 30 --out_classes 2 --train_domains 0.0 0.10 --test_domains 1.0 --n_runs 3  > slab_temp/perf.txt

# python  test.py --test_metric acc --dataset slab --model_name slab --method_name perf_match --penalty_ws 1.0 --out_classes 2 --train_domains 0.0 0.10 --n_runs 3  --acc_data_case train

# python test_slab.py --method_name perf_match --penalty_ws 1.0 --n_runs 3 --train_domains 0.0 0.10 --test_domain 0.3

# # python  test.py --test_metric mia --dataset slab --model_name slab --method_name perf_match --penalty_ws 10.0 --mia_logit 1 --mia_sample_size 400 --out_classes 2 --train_domains 0.0 0.10 --test_domains 0.50 --n_runs 3  > slab_temp/perf-mi.txt

# # python  test.py --test_metric privacy_entropy --dataset slab --model_name slab --method_name perf_match --penalty_ws 10.0 --mia_sample_size 400 --out_classes 2 --train_domains 0.0 0.10 --test_domains 0.50 --n_runs 3  > slab_temp/perf-entropy.txt

# # python  test.py --test_metric privacy_loss_attack --dataset slab --model_name slab --method_name perf_match --penalty_ws 10.0 --mia_sample_size 400 --out_classes 2 --train_domains 0.0 0.10 --test_domains 0.50 --n_runs 3



# #IRM
# python train.py --dataset slab --model_name slab --method_name irm_slab --batch_size 64 --lr 0.1 --penalty_irm 10.0 --penalty_s 2 --epochs 30 --out_classes 2 --train_domains 0.0 0.10 --test_domains 1.0 --n_runs 3 > slab_temp/irm.txt


# #CSD
# python train.py --dataset slab --model_name slab --method_name csd_slab --batch_size 64 --lr 0.1 --penalty_ws 0.0 --epochs 30 --out_classes 2 --train_domains 0.0 0.10 --test_domains 1.0 --n_runs 3 --rep_dim 100  > slab_temp/csd.txt






# ## Slab Spur
# python train.py --dataset slab_spur --model_name slab --method_name perf_match --batch_size 64 --lr 0.1 --penalty_ws 0.0 --epochs 30 --out_classes 2 --train_domains 0.0 0.10 --test_domains 0.90 --n_runs 3 --slab_data_dim 3

# python train.py --dataset slab_spur --model_name slab --method_name perf_match --batch_size 64 --lr 0.1 --penalty_ws 50.0 --epochs 30 --out_classes 2 --train_domains 0.0 0.10 --test_domains 0.90 --n_runs 3 --slab_data_dim 3

# python train.py --dataset slab_spur --model_name slab --method_name rand_match --batch_size 64 --lr 0.1 --penalty_ws 50.0 --epochs 30 --out_classes 2 --train_domains 0.0 0.10 --test_domains 1.0 --n_runs 3  --slab_data_dim 3

# python train.py --dataset slab_spur --model_name slab --method_name irm_slab --batch_size 64 --lr 0.1 --penalty_irm 10.0 --penalty_s 2 --epochs 30 --out_classes 2 --train_domains 0.0 0.10 --test_domains 1.0 --n_runs 3  --slab_data_dim 3 

# python train.py --dataset slab_spur --model_name slab --method_name csd_slab --batch_size 64 --lr 0.1 --penalty_ws 0.0 --epochs 30 --out_classes 2 --train_domains 0.0 0.10 --test_domains 1.0 --n_runs 3 --rep_dim 100  --slab_data_dim 3 


# # Train Case
# python  test.py --test_metric acc --dataset slab_spur --model_name slab --method_name perf_match --penalty_ws 50.0 --out_classes 2 --train_domains 0.0 0.10 --n_runs 3  --acc_data_case train --slab_data_dim 3

# # #Attribute Attack
# # python test.py --test_metric attribute_attack --mia_logit 1 --attribute_domain 0 --batch_size 64 --dataset slab_spur --model_name slab --method_name perf_match --penalty_ws 0.0 --out_classes 2 --train_domains 0.0 0.10 --test_domains 0.90 --n_runs 3 --slab_data_dim 3 

# python test.py --test_metric attribute_attack --mia_logit 1 --attribute_domain 0 --batch_size 64 --dataset slab_spur --model_name slab --method_name perf_match --penalty_ws 0.0 --out_classes 2 --train_domains 0.0 0.10 --test_domains 0.90 --n_runs 3 --slab_data_dim 3

# python test.py --test_metric attribute_attack --mia_logit 1 --attribute_domain 0 --batch_size 64 --dataset slab_spur --model_name slab --method_name perf_match --penalty_ws 50.0 --out_classes 2 --train_domains 0.0 0.10 --test_domains 0.90 --n_runs 3 --slab_data_dim 3


# python3 slab-run.py train
# python3 slab-run.py test
# python3 slab-run.py train_plot

python test_slab.py --method_name perf_match --penalty_ws 0.0 --n_runs 3 --train_domains 0.0 0.10 --test_domain 0.90 --slab_noise 0.05 > slab_noise/erm_auc.txt

python test_slab.py --method_name perf_match --penalty_ws 1.0 --n_runs 3 --train_domains 0.0 0.10 --test_domain 0.90 --slab_noise 0.05 > slab_noise/perf_auc.txt

python  test.py --test_metric privacy_entropy --dataset slab --model_name slab --method_name perf_match --penalty_ws 0.0 --mia_sample_size 400 --out_classes 2 --train_domains 0.0 0.10 --test_domains 0.90 --n_runs 3 --slab_noise 0.05 > slab_noise/erm_entropy.txt

python  test.py --test_metric privacy_entropy --dataset slab --model_name slab --method_name perf_match --penalty_ws 1.0 --mia_sample_size 400 --out_classes 2 --train_domains 0.0 0.10 --test_domains 0.90 --n_runs 3 --slab_noise 0.05 > slab_noise/perf_entropy.txt

python  test.py --test_metric privacy_loss_attack --dataset slab --model_name slab --method_name perf_match --penalty_ws 0.0 --mia_sample_size 400 --out_classes 2 --train_domains 0.0 0.10 --test_domains 0.90 --n_runs 3 --slab_noise 0.05 > slab_noise/erm_loss.txt

python  test.py --test_metric privacy_loss_attack --dataset slab --model_name slab --method_name perf_match --penalty_ws 1.0 --mia_sample_size 400 --out_classes 2 --train_domains 0.0 0.10 --test_domains 0.90 --n_runs 3 --slab_noise 0.05 > slab_noise/perf_loss.txt

# python  test.py --test_metric mia --dataset slab --model_name slab --method_name perf_match --penalty_ws 0.0 --mia_logit 1 --mia_sample_size 400 --out_classes 2 --train_domains 0.0 0.10 --test_domains 0.15 --n_runs 3 --slab_noise 0.05 > slab_noise/erm_mi.txt

# python  test.py --test_metric mia --dataset slab --model_name slab --method_name perf_match --penalty_ws 1.0 --mia_logit 1 --mia_sample_size 400 --out_classes 2 --train_domains 0.0 0.10 --test_domains 0.15 --n_runs 3 --slab_noise 0.05 > slab_noise/perf_mi.txt



python train.py --dataset slab --model_name slab --method_name perf_match --batch_size 64 --lr 0.1 --penalty_ws 0.0 --epochs 30 --out_classes 2 --train_domains 0.0 0.10 --test_domains 0.90 --n_runs 3 --slab_noise 0.20    

python train.py --dataset slab --model_name slab --method_name mask_linear --batch_size 64 --lr 0.1 --penalty_ws 0.0 --epochs 30 --out_classes 2 --train_domains 0.0 0.10 --test_domains 0.90 --n_runs 3 --slab_noise 0.20

python test_slab.py --method_name perf_match --penalty_ws 0.0 --n_runs 3 --train_domains 0.0 0.10 --test_domain 0.30 --slab_noise 0.20

python test_slab.py --method_name mask_linear --penalty_ws 0.0 --n_runs 3 --train_domains 0.0 0.10 --test_domain 0.3 --slab_noise 0.20

python  test.py --test_metric mia --dataset slab --model_name slab --method_name perf_match --penalty_ws 0.0 --mia_logit 1 --mia_sample_size 400 --out_classes 2 --train_domains 0.0 0.10 --test_domains 0.30 --n_runs 3 --slab_noise 0.20

 python  test.py --test_metric mia --dataset slab --model_name slab --method_name mask_linear --penalty_ws 0.0 --mia_logit 1 --mia_sample_size 400 --out_classes 2 --train_domains 0.0 0.10 --test_domains 0.30 --n_runs 3 --slab_noise 0.20