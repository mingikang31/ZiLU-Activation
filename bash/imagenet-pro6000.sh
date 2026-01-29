#! /bin/bash 
#SBATCH --nodes=1 
#SBATCH --mem=480G
#SBATCH -p mixed --gres=gpu:pro6000:1
#SBATCH --cpus-per-gpu=80
#SBATCH --job-name=imagenet1k_exp
#SBATCH --time=720:00:00
#SBATCH --output=slurm_out/%j.out
#SBATCH --error=slurm_out/%j.err
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80
#SBATCH --mail-user=mkang2@bowdoin.edu

source ~/.bashrc
conda activate torch-pro6000

cd /mnt/research/j.farias/mkang2/ZiLU-Activation 

python vision_main.py \
    --model resnet50 \
    --dataset imagenet1k \
    --compile \
    --use_amp \
    --batch_size 1024 \
    --device cuda \
    --output_dir ./Output/ImageNet1K/ResNet50/test-1 \
    --num_workers 12 \
    --pin_memory