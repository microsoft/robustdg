# python  test.py --test_metric privacy_entropy --mia_sample_size 1000 --batch_size 64 --dataset pacs  --method_name erm_match --match_case 0.01 --penalty_ws 0 --out_classes 7 --perfect_match 0 --img_c 3 --pre_trained 1 --model_name resnet18 --train_domains art_painting cartoon sketch --test_domains photo > results/pacs/entropy/photo_erm.txt

# python  test.py --test_metric privacy_entropy --mia_sample_size 1000 --batch_size 64 --dataset pacs  --method_name erm_match --match_case 0.01 --penalty_ws 0 --out_classes 7 --perfect_match 0 --img_c 3 --pre_trained 1 --model_name resnet18 --train_domains photo cartoon sketch --test_domains art_painting > results/pacs/entropy/painting_erm.txt

# python  test.py --test_metric privacy_entropy --mia_sample_size 1000 --batch_size 64 --dataset pacs  --method_name erm_match --match_case 0.01 --penalty_ws 0 --out_classes 7 --perfect_match 0 --img_c 3 --pre_trained 1 --model_name resnet18 --train_domains photo art_painting sketch --test_domains cartoon > results/pacs/entropy/cartoon_erm.txt

# python  test.py --test_metric privacy_entropy --mia_sample_size 1000 --batch_size 64 --dataset pacs  --method_name erm_match --match_case 0.01 --penalty_ws 0 --out_classes 7 --perfect_match 0 --img_c 3 --pre_trained 1 --model_name resnet18 --train_domains photo art_painting cartoon --test_domains sketch > results/pacs/entropy/sketch_erm.txt


# python  test.py --test_metric privacy_entropy --mia_sample_size 1000 --batch_size 64 --dataset pacs  --method_name erm_match --match_case 0.01 --penalty_ws 0.1 --out_classes 7 --perfect_match 0 --img_c 3 --pre_trained 1 --model_name resnet18 --train_domains art_painting cartoon sketch --test_domains photo > results/pacs/entropy/photo_random.txt

# python  test.py --test_metric privacy_entropy --mia_sample_size 1000 --batch_size 64 --dataset pacs  --method_name erm_match --match_case 0.01 --penalty_ws 0.5 --out_classes 7 --perfect_match 0 --img_c 3 --pre_trained 1 --model_name resnet18 --train_domains photo cartoon sketch --test_domains art_painting > results/pacs/entropy/painting_random.txt

# python  test.py --test_metric privacy_entropy --mia_sample_size 1000 --batch_size 64 --dataset pacs  --method_name erm_match --match_case 0.01 --penalty_ws 0.1 --out_classes 7 --perfect_match 0 --img_c 3 --pre_trained 1 --model_name resnet18 --train_domains photo art_painting sketch --test_domains cartoon > results/pacs/entropy/cartoon_random.txt

# python  test.py --test_metric privacy_entropy --mia_sample_size 1000 --batch_size 64 --dataset pacs  --method_name erm_match --match_case 0.01 --penalty_ws 0.1 --out_classes 7 --perfect_match 0 --img_c 3 --pre_trained 1 --model_name resnet18 --train_domains photo art_painting cartoon --test_domains sketch > results/pacs/entropy/sketch_random.txt

python test.py --test_metric privacy_entropy --mia_sample_size 1000 --batch_size 64 --dataset pacs --method_name hybrid --ctr_match_case 0.01 --ctr_match_flag 1 --ctr_match_interrupt 5 --ctr_model_name resnet18 --out_classes 7 --perfect_match 0 --img_c 3 --pre_trained 1 --model_name resnet18 --penalty_ws 0.1 --match_case -1 --train_domains art_painting cartoon sketch --test_domains photo >  results/pacs/entropy/photo_hybrid.txt

python test.py --test_metric privacy_entropy --mia_sample_size 1000 --batch_size 64 --dataset pacs --method_name hybrid --ctr_match_case 0.01 --ctr_match_flag 1 --ctr_match_interrupt 5 --ctr_model_name resnet18 --out_classes 7 --perfect_match 0 --img_c 3 --pre_trained 1 --model_name resnet18 --penalty_ws 0.01 --match_case -1 --train_domains photo cartoon sketch --test_domains art_painting > results/pacs/entropy/painting_hybrid.txt

python test.py --test_metric privacy_entropy --mia_sample_size 1000 --batch_size 64 --dataset pacs --method_name hybrid --ctr_match_case 0.01 --ctr_match_flag 1 --ctr_match_interrupt 5 --ctr_model_name resnet18 --out_classes 7 --perfect_match 0 --img_c 3 --pre_trained 1 --model_name resnet18 --penalty_ws 0.01 --match_case -1 --train_domains photo art_painting sketch --test_domains cartoon > results/pacs/entropy/cartoon_hybrid.txt

python test.py --test_metric privacy_entropy --mia_sample_size 1000 --batch_size 64 --dataset pacs --method_name hybrid --ctr_match_case 0.01 --ctr_match_flag 1 --ctr_match_interrupt 5 --ctr_model_name resnet18 --out_classes 7 --perfect_match 0 --img_c 3 --pre_trained 1 --model_name resnet18 --penalty_ws 0.5 --match_case -1 --train_domains photo art_painting cartoon --test_domains sketch > results/pacs/entropy/sketch_hybrid.txt