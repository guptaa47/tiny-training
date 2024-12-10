#!/bin/bash

# Load required modules
module load anaconda/Python-ML-2024b
# module load cuda/11.8

# SLURM directives (must start at the beginning of the line with no extra spaces)
#SBATCH -n 20
#SBATCH -N 1
#SBATCH --gres=gpu:volta:1

# here we compare transfer learning accuracy w/ and w/o qas, as well as for sparse updates
# we update the last 6 layers to simulate partial update

model="mcunet-5fps"
dataset="mini-person"

# # w/o qas:
# python algorithm/train_cls.py algorithm/configs/transfer.yaml --run_dir algorithm/runs/${dataset}/${model}/6b+6w/sgd \
#     --net_name ${model}  --bs256_lr  0.1  --optimizer_name sgd \
#     --enable_backward_config 1 --n_bias_update 6 --n_weight_update 6

# # w/ qas:
# python algorithm/train_cls.py algorithm/configs/transfer.yaml --run_dir algorithm/runs/${dataset}/${model}/6b+6w/sgd_qas \
#     --net_name ${model}  --bs256_lr  0.075  --optimizer_name sgd_scale \
#     --enable_backward_config 1 --n_bias_update 6 --n_weight_update 6

# # sparse update (mbv2, 50KB scheme):
# python algorithm/train_cls.py algorithm/configs/transfer.yaml --run_dir algorithm/runs/${dataset}/${model}/sparse_50kb/sgd_qas_nomom \
#     --net_name ${model}  --bs256_lr  0.1  --optimizer_name sgd_scale_nomom \
#     --enable_backward_config 1 --n_bias_update 16 --manual_weight_idx 36-39-40-41-42-45-48-49 \
#     --weight_update_ratio 1-1-0-0.25-0.125-0.125-0.125-0.125

# sparse update (mcunet, 50KB scheme):
python algorithm/train_cls.py algorithm/configs/transfer.yaml --run_dir algorithm/runs/${dataset}/${model}/sparse_50kb/sgd_qas_nomom \
    --net_name ${model}  --bs256_lr  0.1  --optimizer_name sgd_scale_nomom \
    --enable_backward_config 1 --n_bias_update 20 --manual_weight_idx 23-24-27-30-33-39 \
    --weight_update_ratio 0-0.25-0.5-0.5-0-0

# # sparse update (proxyless, 50KB scheme):
# python algorithm/train_cls.py algorithm/configs/transfer.yaml --run_dir algorithm/runs/${dataset}/${model}/sparse_50kb/sgd_qas_nomom \
#     --net_name ${model}  --bs256_lr  0.1  --optimizer_name sgd_scale_nomom \
#     --enable_backward_config 1 --n_bias_update 21 --manual_weight_idx 39-42-44-45-50-51-54-57 \
#     --weight_update_ratio 0.25-1-0-1-0-0.125-0.25-0.25

# # update last 12 weights and biases:
# python algorithm/train_cls.py algorithm/configs/transfer.yaml --run_dir algorithm/runs/${dataset}/${model}/last_12/sgd_qas_nomom \
#     --net_name ${model}  --bs256_lr  0.1  --optimizer_name sgd_scale_nomom \
#     --enable_backward_config 1 --n_bias_update 12 --n_weight_update 12

# # update full network:
# python algorithm/train_cls.py algorithm/configs/transfer.yaml --run_dir algorithm/runs/${dataset}/${model}/full_update/sgd_qas_nomom \
#     --net_name ${model}  --bs256_lr  0.1  --optimizer_name sgd_scale_nomom