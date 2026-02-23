#!/bin/bash 
#SBATCH --nodes=1 
#SBATCH --mem=128G
#SBATCH -p arm --gres=shard:32
#SBATCH --cpus-per-task=48
#SBATCH --job-name=no_norm
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

DATASETS=("cifar10")
ACTIVATIONS=('relu' 'gelu' 'zilu')
LR="1e-3"
for ds in "${DATASETS[@]}"; do 
    for act in "${ACTIVATIONS[@]}"; do 

        COUNT=$((COUNT + 1)) 

        output_dir="./Output/AUG/No-Norm_ViT-Tiny/$(echo $ds | awk '{print toupper($0)}')/${act}_s42" 

        echo "[$COUNT] Dataset=$ds | Activation=$act"

        python vision_main.py \
            --activation $act \
            --sigma 1.0 \
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
            --pin_memory

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

DATASETS=("cifar10")
ACTIVATIONS=('relu' 'gelu' 'zilu')
LR="1e-3"
for ds in "${DATASETS[@]}"; do 
    for act in "${ACTIVATIONS[@]}"; do 

        COUNT=$((COUNT + 1)) 

        output_dir="./Output/AUG/No-Norm_ResNet34/$(echo $ds | awk '{print toupper($0)}')/${act}_s42" 

        echo "[$COUNT] Dataset=$ds | Activation=$act"

        python vision_main.py \
            --activation $act \
            --sigma 1.0 \
            --inplace \
            --model resnet34 \
            --dataset $ds \
            --augment \
            --data_path ./Data \
            --batch_size 128 \
            --num_epochs 200 \
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
            --pin_memory

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


echo "=========================================="
echo "Completed No Norm Experiments"
echo "------------------------------------------"
echo "Total experiments: $COUNT"
echo "Successful: $((COUNT - FAILED))"
echo "Failed: $FAILED"
echo "=========================================="
