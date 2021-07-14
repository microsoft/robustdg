#Photo
python train.py --dataset pacs --method_name matchdg_ctr --match_case 0.01 --match_flag 1 --epochs 15 --batch_size 128 --pos_metric cos --train_domains art_painting cartoon sketch --test_domains photo --out_classes 7 --perfect_match 0 --img_c 3 --pre_trained 1 --model_name $1


#Art Painting
python train.py --dataset pacs --method_name matchdg_ctr --match_case 0.01 --match_flag 1 --epochs 15 --batch_size 128 --pos_metric cos --train_domains photo cartoon sketch --test_domains art_painting --out_classes 7 --perfect_match 0 --img_c 3 --pre_trained 1 --model_name $1


#Cartoon
python train.py --dataset pacs --method_name matchdg_ctr --match_case 0.01 --match_flag 1 --epochs 15 --batch_size 128 --pos_metric cos --train_domains photo art_painting sketch --test_domains cartoon --out_classes 7 --perfect_match 0 --img_c 3 --pre_trained 1 --model_name $1


#Sketch
python train.py --dataset pacs --method_name matchdg_ctr --match_case 0.01 --match_flag 1 --epochs 15 --batch_size 128 --pos_metric cos --train_domains photo art_painting cartoon --test_domains sketch --out_classes 7 --perfect_match 0 --img_c 3 --pre_trained 1 --model_name $1

