# GradMask: Gradient-Guided Token Masking for Textual Adversarial Example Detection

Accepted in 28th SIGKDD Conference on Knowledge Discovery and Data Mining.

## Command
Train a classifier:
```
CUDA_VISIBLE_DEVICES=0 python3 main_org.py --model_dir_path ./cls_task/checkpoint/ --dataset imdb --lr 0.00001 --batch_size 16 --epochs 10 --save_model cls_rob_large_imdb_test --epochs 10 --model roberta
```

Attack the trained model:
```
CUDA_VISIBLE_DEVICES=0 python3 textattack_attack_orig.py --model_dir_path ./cls_task/checkpoint/ --load_model cls_roberta_ag_3 --p_vocab 1.0 --dataset ag --nth_data 0 --seed 0 --dataset_type test --save_data True --model roberta --attack_method textfooler --n_success 1000
```

Detect adversarial examples:
```
CUDA_VISIBLE_DEVICES=0 python3 main_adv_det.py --model_dir_path ./cls_task/checkpoint/ --dataset mr --batch_size 16 --load_model cls_rob_mr_0 --model roberta --multi_mask 1 --attack_method pwws --conf_feature conf_sub_square --det_mode grad --dataset_type test
```


