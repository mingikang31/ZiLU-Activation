#! /bin/bash 
#SBATCH --nodes=1 
#SBATCH --mem=256G
#SBATCH -p gpu --gres=gpu:rtx3080:3
#SBATCH --cpus-per-gpu=1
#SBATCH --job-name=multi_gpu_test
#SBATCH --time=720:00:00
#SBATCH --output=slurm_out/%j.out
#SBATCH --error=slurm_out/%j.err
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80
#SBATCH --mail-user=mkang2@bowdoin.edu

source ~/.bashrc
conda activate torch-rtx3080

cd /mnt/research/j.farias/mkang2/ZiLU-Activation 

# --- Safe Port Generation --- # 
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
echo "Using Port: $MASTER_PORT"

torchrun --nproc_per_node=3 \
         --rdzv_backend=c10d \
         --rdzv_endpoint=localhost:$MASTER_PORT \
         vision_main.py \
        --model resnet50 \
        --dataset imagenet1k \
        --compile \
        --use_amp \
        --device cuda \
        --output_dir ./Output/ImageNet1K/ResNet50/multigpu-test-2 \
        --num_workers 1 \
        --pin_memory \
        --ddp \
        --ddp_batch_size 128

