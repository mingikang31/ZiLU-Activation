#!/bin/bash 
#SBATCH --nodes=1 
#SBATCH --mem=128G
#SBATCH -p arm --gres=shard:32
#SBATCH --cpus-per-task=48
#SBATCH --job-name=resnet_exp
#SBATCH --time=96:00:00
#SBATCH --output=slurm_out/%j.out
#SBATCH --error=slurm_out/%j.err
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80
#SBATCH --mail-user=mkang2@bowdoin.edu

source ~/.bashrc
conda activate torch-gh200

cd /mnt/research/j.farias/mkang2/ZiLU-Activation 


COUNT=0
FAILED=0

IMBFACTOR="0.01"

### ResNet34
DATASETS=("cifar100-lt")
ACTIVATIONS=('relu' 'gelu' 'silu')
LR="1e-3"
# Baseline Experiments (ReLU, GeLU, SiLU) 
for ds in "${DATASETS[@]}"; do 
    for act in "${ACTIVATIONS[@]}"; do 

        COUNT=$((COUNT + 1)) 

        output_dir="./Output/AUG/ResNet34/$(echo $ds | awk '{print toupper($0)}')/${IMBFACTOR}/${act}_s42" 

        echo "[$COUNT] Dataset=$ds | Activation=$act"

        python vision_main.py \
            --activation $act \
            --inplace \
            --model resnet34 \
            --dataset $ds \
            --augment \
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
            --output_dir $output_dir \
            --num_workers 12 \
            --pin_memory \
            --imb_factor $IMBFACTOR

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
DATASETS=("cifar100-lt")
ACTIVATIONS=('zilu' 'zilu_approx')
LR="1e-3"
SIGMAS=("0.1" "0.5" "1.0" "5.0" "10.0")

# Vary sigmas
for ds in "${DATASETS[@]}"; do 
    for act in "${ACTIVATIONS[@]}"; do 
        for sigma in "${SIGMAS[@]}"; do 

            COUNT=$((COUNT + 1)) 

            output_dir="./Output/AUG/ResNet34/$(echo $ds | awk '{print toupper($0)}')/${IMBFACTOR}/${act}_sigma${sigma}_s42"

            echo "[$COUNT] Dataset=$ds | Activation=$act | Sigma=$sigma"

            python vision_main.py \
                --activation $act \
                --sigma $sigma \
                --inplace \
                --model resnet34 \
                --dataset $ds \
                --augment \
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
                --output_dir $output_dir \
                --num_workers 12 \
                --pin_memory \
                --imb_factor $IMBFACTOR

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
done 

### ViT-Tiny
DATASETS=("cifar100-lt")
ACTIVATIONS=('relu' 'gelu' 'silu')
LR="1e-3"
# Baseline Experiments (ReLU, GeLU, SiLU) 
for ds in "${DATASETS[@]}"; do 
    for act in "${ACTIVATIONS[@]}"; do 

        COUNT=$((COUNT + 1)) 

        output_dir="./Output/AUG/VIT-Tiny/$(echo $ds | awk '{print toupper($0)}')/${IMBFACTOR}/${act}_s42" 

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
            --output_dir $output_dir \
            --num_workers 12 \
            --pin_memory \
            --imb_factor $IMBFACTOR

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
DATASETS=("cifar100-lt")
ACTIVATIONS=('zilu' 'zilu_approx')
LR="1e-3"
SIGMAS=("0.1" "0.5" "1.0" "5.0" "10.0")

# Vary sigmas
for ds in "${DATASETS[@]}"; do 
    for act in "${ACTIVATIONS[@]}"; do 
        for sigma in "${SIGMAS[@]}"; do 

            COUNT=$((COUNT + 1)) 

            output_dir="./Output/AUG/VIT-Tiny/$(echo $ds | awk '{print toupper($0)}')/${IMBFACTOR}/${act}_sigma${sigma}_s42"

            echo "[$COUNT] Dataset=$ds | Activation=$act | Sigma=$sigma"

            python vision_main.py \
                --activation $act \
                --sigma $sigma \
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
                --output_dir $output_dir \
                --num_workers 12 \
                --pin_memory \
                --imb_factor $IMBFACTOR

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
done 



echo "=========================================="
echo "Completed CIFAR100 Long Tailed ResNet34 & ViT-Tiny Experiments"
echo "------------------------------------------"
echo "Total experiments: $COUNT"
echo "Successful: $((COUNT - FAILED))"
echo "Failed: $FAILED"
echo "=========================================="


