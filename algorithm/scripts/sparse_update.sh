#!/bin/bash

# Load required modules
module load anaconda/Python-ML-2024b
# module load cuda/11.8

# SLURM directives (must start at the beginning of the line with no extra spaces)
#SBATCH -n 20
#SBATCH -N 1
#SBATCH --gres=gpu:volta:1

# our sparse update can achieve higher accuracy at lower memory usage
# here we use mcunet-5fps as an  example
# we use optimizer without momentum which saves memory (as in the main results of the paper)

model="mcunet-5fps"

# sparse update (50KB scheme):
python algorithm/train_cls.py algorithm/configs/transfer.yaml --run_dir algorithm/runs/flowers/${model}/sparse_49kb/sgd_qas_nomom \
    --net_name ${model}  --bs256_lr  0.1  --optimizer_name sgd_scale_nomom \
    --enable_backward_config 1 --n_bias_update 22 --manual_weight_idx 23-24-27-30-33-39 \
    --weight_update_ratio 0-0.25-0.5-0.5-0-0

# sparse update (100KB scheme):
python algorithm/train_cls.py algorithm/configs/transfer.yaml --run_dir algorithm/runs/flowers/${model}/sparse_100kb/sgd_qas_nomom \
    --net_name ${model}  --bs256_lr  0.1  --optimizer_name sgd_scale_nomom \
    --enable_backward_config 1 --n_bias_update 22 --manual_weight_idx 21-24-27-30-36-39 \
    --weight_update_ratio 1-1-1-1-0.125-0.25

# update last 12 weights and biases:
python algorithm/train_cls.py algorithm/configs/transfer.yaml --run_dir algorithm/runs/flowers/${model}/last_12/sgd_qas_nomom \
    --net_name ${model}  --bs256_lr  0.1  --optimizer_name sgd_scale_nomom \
    --enable_backward_config 1 --n_bias_update 12 --n_weight_update 12

# update full network:
python algorithm/train_cls.py algorithm/configs/transfer.yaml --run_dir algorithm/runs/flowers/${model}/full_update/sgd_qas_nomom \
    --net_name ${model}  --bs256_lr  0.1  --optimizer_name sgd_scale_nomom