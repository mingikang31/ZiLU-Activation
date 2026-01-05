#! /bin/bash 
#SBATCH --nodes=1 
#SBATCH --mem=64G
#SBATCH -p gpu --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=gpt2_exp
#SBATCH --time=500:00:00
#SBATCH --output=slurm_out/%j.out
#SBATCH --error=slurm_out/%j.err
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80
#SBATCH --mail-user=mkang2@bowdoin.edu

source ~/.bashrc
conda activate torch-a100


cd /mnt/research/j.farias/mkang2/ZiLU-Activation 

DATASETS=("wikitext103")
ACTIVATIONS=('relu' 'gelu' 'silu' 'sigmoid' 'gelu_s' 'silu_s' 'zilu' 'zilu_approx')
LR="6e-4"

COUNT=0
FAILED=0

export TORCH_FLOAT32_MATMUL_PRECISION=high

# Baseline Experiments (ReLU, GeLU, SiLU) 
for ds in "${DATASETS[@]}"; do 
    for act in "${ACTIVATIONS[@]}"; do 

        COUNT=$((COUNT + 1)) 

        output_dir="./Output/GPT2/$(echo $ds | awk '{print toupper($0)}')/${act}_s42" 

        echo "[$COUNT] Dataset=$ds | Activation=$act"

        python language_main.py \
            --vocab_size 50257 \
            --max_seq_length 1024 \
            --embedding_dim 768 \
            --num_attention_heads 12 \
            --num_layers 12 \
            --activation $act \
            --inplace \
            --dataset $ds \
            --compile \
            --use_amp \
            --data_path ./Data \
            --batch_size 48 \
            --num_epochs 20 \
            --clip_grad_norm 1.0 \
            --optimizer adamw \
            --weight_decay 0.1 \
            --lr $LR \
            --scheduler linear \
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


# Vary Sigmas 
DATASETS=("wikitext103")
ACTIVATIONS=('gelu_s' 'silu_s' 'zilu' 'zilu_approx')
LR="6e-4"
SIGMAS=("0.01" "0.05" "0.1" "0.5" "1.0" "5.0" "10.0" "50.0" "100.0" "500.0" "1000.0")

# Vary sigmas
for ds in "${DATASETS[@]}"; do 
    for act in "${ACTIVATIONS[@]}"; do 
        for sigma in "${SIGMAS[@]}"; do 

            COUNT=$((COUNT + 1)) 

            output_dir="./Output/GPT2/$(echo $ds | awk '{print toupper($0)}')/${act}_sigma${sigma}_s42"

            echo "[$COUNT] Dataset=$ds | Activation=$act | Sigma=$sigma"

            python language_main.py \
                --vocab_size 50257 \
                --max_seq_length 1024 \
                --embedding_dim 768 \
                --num_attention_heads 12 \
                --num_layers 12 \
                --activation $act \
                --sigma $sigma \
                --inplace \
                --dataset $ds \
                --compile \
                --use_amp \
                --data_path ./Data \
                --batch_size 48 \
                --num_epochs 20 \
                --clip_grad_norm 1.0 \
                --optimizer adamw \
                --weight_decay 0.1 \
                --lr $LR \
                --scheduler linear \
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
done 

echo "=========================================="
echo "Completed GPT2 Experiments"
echo "------------------------------------------"
echo "Total experiments: $COUNT"
echo "Successful: $((COUNT - FAILED))"
echo "Failed: $FAILED"
echo "=========================================="
