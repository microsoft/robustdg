#Kaggle
python3 cxray_run.py kaggle acc train
python3 cxray_run.py kaggle acc test
python3 cxray_run.py kaggle privacy_loss_attack
python3 cxray_run.py kaggle privacy_entropy
python3 cxray_run.py kaggle match_score test

#ChexPert
python3 cxray_run.py chex acc train
python3 cxray_run.py chex acc test
python3 cxray_run.py chex privacy_loss_attack
python3 cxray_run.py chex privacy_entropy
python3 cxray_run.py chex match_score test

#NIH
python3 cxray_run.py nih acc train
python3 cxray_run.py nih acc test
python3 cxray_run.py nih privacy_loss_attack
python3 cxray_run.py nih privacy_entropy
python3 cxray_run.py nih match_score test

# #Accuracy Train

# python test.py --test_metric acc --batch_size 64 --dataset chestxray --acc_data_case train --match_func_aug_case 1 --method_name erm_match --match_case 0.01 --penalty_ws 0 --out_classes 2 --perfect_match 0 --img_c 3 --pre_trained 1 --model_name densenet121 --train_domains nih_trans chex_trans --test_domains kaggle > results/chestxray/acc_train/erm.txt   

# python test.py --test_metric acc --batch_size 64 --dataset chestxray --acc_data_case train --match_func_aug_case 1 --method_name erm_match --match_case 0.01 --penalty_ws 10.0 --out_classes 2 --perfect_match 0 --img_c 3 --pre_trained 1 --model_name densenet121 --train_domains nih_trans chex_trans --test_domains kaggle > results/chestxray/acc_train/random.txt

# python test.py --test_metric acc --batch_size 64 --dataset chestxray --acc_data_case train --match_func_aug_case 1 --method_name csd --match_case 0.01 --penalty_ws 0 --out_classes 2 --perfect_match 0 --img_c 3 --pre_trained 1 --model_name densenet121 --train_domains nih_trans chex_trans --test_domains kaggle > results/chestxray/acc_train/csd.txt
 
# python test.py --test_metric acc --batch_size 64 --dataset chestxray --acc_data_case train --match_func_aug_case 1 --method_name irm --match_case 0.01 --penalty_s 5 --penalty_ws 10.0 --out_classes 2 --perfect_match 0 --img_c 3 --pre_trained 1 --model_name densenet121 --train_domains nih_trans chex_trans --test_domains kaggle > results/chestxray/acc_train/irm.txt

# python test.py --test_metric acc --batch_size 64 --dataset chestxray --acc_data_case train --match_func_aug_case 1 --method_name matchdg_erm --ctr_match_case 0.01 --ctr_match_flag 1 --ctr_match_interrupt 5 --ctr_model_name densenet121 --out_classes 2 --perfect_match 0 --img_c 3 --pre_trained 1 --model_name densenet121 --penalty_ws 50.0 --match_case -1 --train_domains nih_trans chex_trans --test_domains kaggle > results/chestxray/acc_train/matchdg.txt

# # #Accuracy Test

# # python test.py --test_metric acc --batch_size 64 --dataset chestxray --acc_data_case test --match_func_aug_case 1 --method_name erm_match --match_case 0.01 --penalty_ws 0 --out_classes 2 --perfect_match 0 --img_c 3 --pre_trained 1 --model_name densenet121 --train_domains nih_trans chex_trans --test_domains kaggle > results/chestxray/acc_test/erm.txt   

# # python test.py --test_metric acc --batch_size 64 --dataset chestxray --acc_data_case test --match_func_aug_case 1 --method_name erm_match --match_case 0.01 --penalty_ws 10.0 --out_classes 2 --perfect_match 0 --img_c 3 --pre_trained 1 --model_name densenet121 --train_domains nih_trans chex_trans --test_domains kaggle > results/chestxray/acc_test/random.txt

# # python test.py --test_metric acc --batch_size 64 --dataset chestxray --acc_data_case test --match_func_aug_case 1 --method_name csd --match_case 0.01 --penalty_ws 0 --out_classes 2 --perfect_match 0 --img_c 3 --pre_trained 1 --model_name densenet121 --train_domains nih_trans chex_trans --test_domains kaggle > results/chestxray/acc_test/csd.txt
 
# # python test.py --test_metric acc --batch_size 64 --dataset chestxray --acc_data_case test --match_func_aug_case 1 --method_name irm --match_case 0.01 --penalty_s 5 --penalty_ws 10.0 --out_classes 2 --perfect_match 0 --img_c 3 --pre_trained 1 --model_name densenet121 --train_domains nih_trans chex_trans --test_domains kaggle > results/chestxray/acc_test/irm.txt

# # python test.py --test_metric acc --batch_size 64 --dataset chestxray --acc_data_case test --match_func_aug_case 1 --method_name matchdg_erm --ctr_match_case 0.01 --ctr_match_flag 1 --ctr_match_interrupt 5 --ctr_model_name densenet121 --out_classes 2 --perfect_match 0 --img_c 3 --pre_trained 1 --model_name densenet121 --penalty_ws 50.0 --match_case -1 --train_domains nih_trans chex_trans --test_domains kaggle > results/chestxray/acc_test/matchdg.txt

# # #Mean Rank

# # python test.py --test_metric match_score --batch_size 64 --dataset chestxray --match_func_data_case test --match_func_aug_case 1 --method_name erm_match --match_case 0.01 --penalty_ws 0 --out_classes 2 --perfect_match 0 --img_c 3 --pre_trained 1 --model_name densenet121 --train_domains nih_trans chex_trans --test_domains kaggle > results/chestxray/match_score/erm.txt   

# # python test.py --test_metric match_score --batch_size 64 --dataset chestxray --match_func_data_case test --match_func_aug_case 1 --method_name erm_match --match_case 0.01 --penalty_ws 10.0 --out_classes 2 --perfect_match 0 --img_c 3 --pre_trained 1 --model_name densenet121 --train_domains nih_trans chex_trans --test_domains kaggle 

# # python test.py --test_metric match_score --batch_size 64 --dataset chestxray --match_func_data_case test --match_func_aug_case 1 --method_name csd --match_case 0.01 --penalty_ws 0 --out_classes 2 --perfect_match 0 --img_c 3 --pre_trained 1 --model_name densenet121 --train_domains nih_trans chex_trans --test_domains kaggle
 
# # python test.py --test_metric match_score --batch_size 64 --dataset chestxray --match_func_data_case test --match_func_aug_case 1 --method_name irm --match_case 0.01 --penalty_s 5 --penalty_ws 10.0 --out_classes 2 --perfect_match 0 --img_c 3 --pre_trained 1 --model_name densenet121 --train_domains nih_trans chex_trans --test_domains kaggle

# # python test.py --test_metric match_score --batch_size 64 --dataset chestxray --match_func_data_case test --match_func_aug_case 1 --method_name matchdg_erm --ctr_match_case 0.01 --ctr_match_flag 1 --ctr_match_interrupt 5 --ctr_model_name densenet121 --out_classes 2 --perfect_match 0 --img_c 3 --pre_trained 1 --model_name densenet121 --penalty_ws 50.0 --match_case -1 --train_domains nih_trans chex_trans --test_domains kaggle

# # #MIA Attack

# # python  test.py --test_metric mia --mia_sample_size 1000 --mia_logit 1 --batch_size 64 --dataset chestxray --method_name erm_match --match_case 0.01 --penalty_ws 0 --out_classes 2 --perfect_match 0 --img_c 3 --pre_trained 1 --model_name densenet121 --train_domains nih_trans chex_trans --test_domains kaggle > results/chestxray/privacy/erm.txt 

# # python  test.py --test_metric mia --mia_sample_size 1000 --batch_size 64 --dataset chestxray --method_name erm_match --match_case 0.01 --penalty_ws 10 --out_classes 2 --perfect_match 0 --img_c 3 --pre_trained 1 --model_name densenet121 --train_domains nih_trans chex_trans --test_domains kaggle > results/chestxray/privacy/random.txt

# # python test.py --test_metric mia --mia_sample_size 1000 --batch_size 64 --dataset chestxray --method_name csd --match_case 0.01 --penalty_ws 0 --out_classes 2 --perfect_match 0 --img_c 3 --pre_trained 1 --model_name densenet121 --train_domains nih_trans chex_trans --test_domains kaggle > results/chestxray/privacy/csd.txt

# # python test.py --test_metric mia --mia_sample_size 1000 --batch_size 64 --dataset chestxray --method_name irm --match_case 0.01 --penalty_s 5 --penalty_ws 10.0 --out_classes 2 --perfect_match 0 --img_c 3 --pre_trained 1 --model_name densenet121 --train_domains nih_trans chex_trans --test_domains kaggle > results/chestxray/privacy/irm.txt

# # python test.py --test_metric mia --mia_sample_size 1000 --batch_size 64 --dataset chestxray --method_name matchdg_erm --ctr_match_case 0.01 --ctr_match_flag 1 --ctr_match_interrupt 5 --ctr_model_name densenet121 --out_classes 2 --perfect_match 0 --img_c 3 --pre_trained 1 --model_name densenet121 --penalty_ws 50.0 --match_case -1 --train_domains nih_trans chex_trans --test_domains kaggle > results/chestxray/privacy/matchdg.txt

# # #Entorpy Attack

# # python  test.py --test_metric privacy_entropy --mia_sample_size 1000 --batch_size 64 --dataset chestxray --method_name erm_match --match_case 0.01 --penalty_ws 0 --out_classes 2 --perfect_match 0 --img_c 3 --pre_trained 1 --model_name densenet121 --train_domains nih_trans chex_trans --test_domains kaggle > results/chestxray/privacy_entropy/erm.txt

# # python  test.py --test_metric privacy_entropy --mia_sample_size 1000 --batch_size 64 --dataset chestxray --method_name erm_match --match_case 0.01 --penalty_ws 10 --out_classes 2 --perfect_match 0 --img_c 3 --pre_trained 1 --model_name densenet121 --train_domains nih_trans chex_trans --test_domains kaggle > results/chestxray/privacy_entropy/random.txt

# # python test.py --test_metric privacy_entropy --mia_sample_size 1000 --batch_size 64 --dataset chestxray --method_name csd --match_case 0.01 --penalty_ws 0 --out_classes 2 --perfect_match 0 --img_c 3 --pre_trained 1 --model_name densenet121 --train_domains nih_trans chex_trans --test_domains kaggle  > results/chestxray/privacy_entropy/csd.txt

# # python test.py --test_metric privacy_entropy --mia_sample_size 1000 --batch_size 64 --dataset chestxray --method_name irm --match_case 0.01 --penalty_s 5 --penalty_ws 10.0 --out_classes 2 --perfect_match 0 --img_c 3 --pre_trained 1 --model_name densenet121 --train_domains nih_trans chex_trans --test_domains kaggle > results/chestxray/privacy_entropy/irm.txt

# # python test.py --test_metric privacy_entropy --mia_sample_size 1000 --batch_size 64 --dataset chestxray --method_name matchdg_erm --ctr_match_case 0.01 --ctr_match_flag 1 --ctr_match_interrupt 5 --ctr_model_name densenet121 --out_classes 2 --perfect_match 0 --img_c 3 --pre_trained 1 --model_name densenet121 --penalty_ws 50.0 --match_case -1 --train_domains nih_trans chex_trans --test_domains kaggle  > results/chestxray/privacy_entropy/matchdg.txt

# # #Loss Attack

# # python  test.py --test_metric privacy_loss_attack --mia_sample_size 1000 --batch_size 64 --dataset chestxray --method_name erm_match --match_case 0.01 --penalty_ws 0 --out_classes 2 --perfect_match 0 --img_c 3 --pre_trained 1 --model_name densenet121 --train_domains nih_trans chex_trans --test_domains kaggle > results/chestxray/privacy_loss/erm.txt

# # python  test.py --test_metric privacy_loss_attack --mia_sample_size 1000 --batch_size 64 --dataset chestxray --method_name erm_match --match_case 0.01 --penalty_ws 10 --out_classes 2 --perfect_match 0 --img_c 3 --pre_trained 1 --model_name densenet121 --train_domains nih_trans chex_trans --test_domains kaggle > results/chestxray/privacy_loss/random.txt

# # python test.py --test_metric privacy_loss_attack --mia_sample_size 1000 --batch_size 64 --dataset chestxray --method_name csd --match_case 0.01 --penalty_ws 0 --out_classes 2 --perfect_match 0 --img_c 3 --pre_trained 1 --model_name densenet121 --train_domains nih_trans chex_trans --test_domains kaggle  > results/chestxray/privacy_loss/csd.txt

# # python test.py --test_metric privacy_loss_attack --mia_sample_size 1000 --batch_size 64 --dataset chestxray --method_name irm --match_case 0.01 --penalty_s 5 --penalty_ws 10.0 --out_classes 2 --perfect_match 0 --img_c 3 --pre_trained 1 --model_name densenet121 --train_domains nih_trans chex_trans --test_domains kaggle > results/chestxray/privacy_loss/irm.txt

# # python test.py --test_metric privacy_loss_attack --mia_sample_size 1000 --batch_size 64 --dataset chestxray --method_name matchdg_erm --ctr_match_case 0.01 --ctr_match_flag 1 --ctr_match_interrupt 5 --ctr_model_name densenet121 --out_classes 2 --perfect_match 0 --img_c 3 --pre_trained 1 --model_name densenet121 --penalty_ws 50.0 --match_case -1 --train_domains nih_trans chex_trans --test_domains kaggle  > results/chestxray/privacy_loss/matchdg.txt

#Attribute Attack

# python test.py --test_metric attribute_attack --mia_logit 1 --batch_size 64 --attribute_domain 1 --dataset chestxray --method_name erm_match --match_case 0.01 --penalty_ws 0 --out_classes 2 --perfect_match 0 --img_c 3 --pre_trained 1 --model_name densenet121 --train_domains nih_trans chex_trans --test_domains kaggle > results/chestxray/attribute_attack/erm.txt

# python test.py --test_metric attribute_attack --mia_logit 1 --batch_size 64 --attribute_domain 1 --dataset chestxray --method_name erm_match --match_case 0.01 --penalty_ws 10.0 --out_classes 2 --perfect_match 0 --img_c 3 --pre_trained 1 --model_name densenet121 --train_domains nih_trans chex_trans --test_domains kaggle > results/chestxray/attribute_attack/random.txt

# python test.py --test_metric attribute_attack --mia_logit 1 --batch_size 64 --attribute_domain 1 --dataset chestxray --method_name csd --match_case 0.01 --penalty_ws 0 --out_classes 2 --perfect_match 0 --img_c 3 --pre_trained 1 --model_name densenet121 --train_domains nih_trans chex_trans --test_domains kaggle > results/chestxray/attribute_attack/csd.txt

# python test.py --test_metric attribute_attack --mia_logit 1 --batch_size 64 --attribute_domain 1 --dataset chestxray --method_name irm --match_case 0.01 --penalty_s 5 --penalty_ws 10.0 --out_classes 2 --perfect_match 0 --img_c 3 --pre_trained 1 --model_name densenet121 --train_domains nih_trans chex_trans --test_domains kaggle > results/chestxray/attribute_attack/irm.txt 

# python test.py --test_metric attribute_attack --mia_logit 1 --batch_size 64 --attribute_domain 1 --dataset chestxray --method_name matchdg_erm --ctr_match_case 0.01 --ctr_match_flag 1 --ctr_match_interrupt 5 --ctr_model_name densenet121 --out_classes 2 --perfect_match 0 --img_c 3 --pre_trained 1 --model_name densenet121 --penalty_ws 50.0 --match_case -1 --train_domains nih_trans chex_trans --test_domains kaggle > results/chestxray/attribute_attack/matchdg.txt