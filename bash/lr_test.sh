#! /bin/bash 
#SBATCH --nodes=1 
#SBATCH --mem=64G
#SBATCH -p gpu --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=lr_test
#SBATCH --time=500:00:00
#SBATCH --output=slurm_out/%j.out
#SBATCH --error=slurm_out/%j.err
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80
#SBATCH --mail-user=mkang2@bowdoin.edu

source ~/.bashrc
conda activate torch-a100 

cd /mnt/research/j.farias/mkang2/ZiLU-Activation

# Experiment counter 
COUNT=0
FAILED=0

# ReLU, GELU, SiLU Experiments
DATASETS=("cifar10" "cifar100")
ACTIVATIONS=('relu' 'gelu' 'silu')
LRS=("1e-5" "1e-4" "1e-3" "1e-2")

for ds in "${DATASETS[@]}"; do 
    for act in "${ACTIVATIONS[@]}"; do 
        for lr in "${LRS[@]}"; do 

            COUNT=$((COUNT + 1)) 

            output_dir="./Output/AUG/LR-ResNet34/$(echo $ds | awk '{print toupper($0)}')/${act}_lr${lr}_s42"

            echo "[$COUNT] Dataset=$ds | Activation=$act | LR=$lr"

            python vision_main.py \
                --activation $act \
                --sigma 0 \
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
                --lr $lr \
                --scheduler none \
                --device cuda \
                --seed 42 \
                --output_dir $output_dir \
                --num_workers 4
                
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





# ZaiLU, ZaiLU-Approx Experiments
DATASETS=("cifar10" "cifar100")
ACTIVATIONS=('zilu' 'zilu_approx')
SIGMAS=("0.1" "0.5" "1.0" "5.0" "10.0")
LRS=("1e-5" "1e-4" "1e-3" "1e-2")



for ds in "${DATASETS[@]}"; do 
    for act in "${ACTIVATIONS[@]}"; do 
        for sigma in "${SIGMAS[@]}"; do
            for lr in "${LRS[@]}"; do 

                COUNT=$((COUNT + 1)) 

                output_dir="./Output/AUG/LR-ResNet34/$(echo $ds | awk '{print toupper($0)}')/${act}_sigma${sigma}_lr${lr}_s42"

                echo "[$COUNT] Dataset=$ds | Activation=$act | LR=$lr"

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
                    --clip_grad_norm 1.0 \
                    --criterion CrossEntropy \
                    --optimizer adamw \
                    --lr $lr \
                    --scheduler none \
                    --device cuda \
                    --seed 42 \
                    --output_dir $output_dir \
                    --num_workers 4
                    
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
done



echo "=========================================="
echo "Completed ResNet34 Experiments"
echo "------------------------------------------"
echo "Total experiments: $COUNT"
echo "Successful: $((COUNT - FAILED))"
echo "Failed: $FAILED"
echo "=========================================="
