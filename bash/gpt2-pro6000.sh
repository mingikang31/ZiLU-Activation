#! /bin/bash 
#SBATCH --nodes=1 
#SBATCH --mem=64G
#SBATCH -p mixed --gres=gpu:pro6000:1
#SBATCH --cpus-per-gpu=64
#SBATCH --job-name=gpt2_exp
#SBATCH --time=500:00:00
#SBATCH --output=slurm_out/%j.out
#SBATCH --error=slurm_out/%j.err
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80
#SBATCH --mail-user=mkang2@bowdoin.edu

source ~/.bashrc
conda activate torch-pro6000

cd /mnt/research/j.farias/mkang2/ZiLU-Activation 

DATASETS=("wikitext103")
ACTIVATIONS=('leaky_relu' 'prelu' 'elu' 'hardshrink' 'softshrink' 'tanhshrink' 'hardtanh' 'softplus' 'softsign' 'tanh' 'celu' 'mish' 'hardswish' 'hardsigmoid' 'selu' 'squareplus')
LR="6e-4"

COUNT=0
FAILED=0


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
            --batch_size 40 \
            --num_epochs 20 \
            --clip_grad_norm 1.0 \
            --optimizer adamw \
            --weight_decay 0.1 \
            --lr $LR \
            --scheduler linear \
            --device cuda \
            --seed 42 \
            --output_dir $output_dir \
            --num_workers 12 \
            --persistent_workers \
            --prefetch_factor 3 \
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
echo "Completed GPT2 Experiments"
echo "------------------------------------------"
echo "Total experiments: $COUNT"
echo "Successful: $((COUNT - FAILED))"
echo "Failed: $FAILED"
echo "=========================================="
