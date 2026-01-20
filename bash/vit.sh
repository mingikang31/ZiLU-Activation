#! /bin/bash 
#SBATCH --nodes=1 
#SBATCH --mem=64G
#SBATCH -p gpu --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=vit_exp
#SBATCH --time=500:00:00
#SBATCH --output=slurm_out/%j.out
#SBATCH --error=slurm_out/%j.err
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80
#SBATCH --mail-user=mkang2@bowdoin.edu

source ~/.bashrc
conda activate torch-a100

cd /mnt/research/j.farias/mkang2/ZiLU-Activation

DATASETS=("cifar10" "cifar100")
# ACTIVATIONS=('relu' 'gelu' 'silu' 'sigmoid' 'gelu_s' 'silu_s' 'zilu_old' 'arctan' 'arctan_approx' 'zilu' 'zilu_approx')
# ACTIVATIONS=('relu' 'gelu' 'silu' 'sigmoid' 'arctan' 'arctan_approx' 'zilu' 'zilu_approx')
# ACTIVATIONS=('gelu_s' 'silu_s')
ACTIVATIONS=('leaky_relu' 'prelu' 'elu' 'hardshrink' 'softshrink' 'tanhshrink' 'hardtanh' 'softplus' 'softsign' 'tanh' 'celu' 'mish' 'hardswish' 'hardsigmoid' 'selu')
LR="1e-3"

COUNT=0
FAILED=0

# Baseline Experiments (ReLU, GeLU, SiLU) 
for ds in "${DATASETS[@]}"; do 
    for act in "${ACTIVATIONS[@]}"; do 

        COUNT=$((COUNT + 1)) 

        output_dir="./Output/AUG/VIT-Tiny/$(echo $ds | awk '{print toupper($0)}')/${act}_s42" 

        echo "[$COUNT] Dataset=$ds | Activation=$act"

        python vision_main.py \
            --activation $act \
            --inplace \
            --model vit-tiny \
            --dataset $ds \
            --augment \
            --resize 224 \
            --data_path ./Data \
            --batch_size 128 \
            --num_epochs 200 \
            --use_amp \
            --clip_grad_norm 1.0 \
            --criterion CrossEntropy \
            --optimizer adamw \
            --weight_decay 1e-2 \
            --lr $LR \
            --scheduler cosine \
            --device cuda \
            --seed 42 \
            --output_dir $output_dir

        # Check if experiment succeeded
        if [ $? -eq 0 ]; then
            echo "✓ Experiment $COUNT succeeded"
        else
            echo "✗ Experiment $COUNT failed"
            FAILED=$((FAILED + 1))
        fi
        echo ""

    done 
done 


# # Vary Sigmas 
# # DATASETS=("cifar10" "cifar100")
# DATASETS=("cifar100")
# # ACTIVATIONS=('gelu_s' 'silu_s' 'zilu_old' 'arctan' 'arctan_approx' 'zilu' 'zilu_approx')
# # ACTIVATIONS=('zilu' 'zilu_approx')
# ACTIVATIONS=('silu_s' 'gelu_s')
# LR="1e-3"
# SIGMAS=("5.0" "10.0" "50.0" "100.0" "500.0" "1000.0" "0.01" "0.05" "0.1" "0.5" "1.0")

# # Vary sigmas
# for ds in "${DATASETS[@]}"; do 
#     for act in "${ACTIVATIONS[@]}"; do 
#         for sigma in "${SIGMAS[@]}"; do 

#             COUNT=$((COUNT + 1)) 

#             output_dir="./Output/AUG/VIT-Tiny/$(echo $ds | awk '{print toupper($0)}')/${act}_sigma${sigma}_s42"

#             echo "[$COUNT] Dataset=$ds | Activation=$act | Sigma=$sigma"

#             python vision_main.py \
#                 --activation $act \
#                 --sigma $sigma \
#                 --inplace \
#                 --model vit-tiny \
#                 --dataset $ds \
#                 --augment \
#                 --resize 224 \
#                 --data_path ./Data \
#                 --batch_size 128 \
#                 --num_epochs 200 \
#                 --use_amp \
# 		--compile \
#                 --clip_grad_norm 1.0 \
#                 --criterion CrossEntropy \
#                 --optimizer adamw \
#                 --weight_decay 1e-2 \
#                 --lr $LR \
#                 --scheduler cosine \
#                 --device cuda \
#                 --seed 42 \
#                 --output_dir $output_dir

#             # Check if experiment succeeded
#             if [ $? -eq 0 ]; then
#                 echo "✓ Experiment $COUNT succeeded"
#             else
#                 echo "✗ Experiment $COUNT failed"
#                 FAILED=$((FAILED + 1))
#             fi
#             echo ""
        
#         done 
#     done 
# done 




echo "=========================================="
echo "Completed VIT-Tiny Experiments"
echo "------------------------------------------"
echo "Total experiments: $COUNT"
echo "Successful: $((COUNT - FAILED))"
echo "Failed: $FAILED"
echo "=========================================="
