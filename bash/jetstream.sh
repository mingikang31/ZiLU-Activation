cd /home/exouser/ZiLU-Activation

ulimit -s unlimited

# --- Safe Port Generation --- # 
export MASTER_PORT=$(shuf -i 10000-65000 -n 1)
echo "Using Port: $MASTER_PORT"

# Supress warning
export OMP_NUM_THREADS=1

torchrun --nproc_per_node=4 \
         --rdzv_backend=c10d \
         --rdzv_endpoint=localhost:$MASTER_PORT \
         vision_main.py \
        --model resnet50 \
        --dataset imagenet1k \
        --data_path /home/exouser/Datasets \
        --compile \
        --use_amp \
        --device cuda \
        --output_dir /home/exouser/ZiLU-Activation/Output/ImageNet1K/ResNet50/jetstream-h100 \
        --num_workers 16 \
        --pin_memory \
        --ddp \
        --ddp_batch_size 1024

### ResNet50 Training Log on JetStream2 H100 g5.4xl ###
# 4x NVIDIA H100 80GB HBM3 GPUs

### gpustat log during training ###
# imagenet1k                Wed Feb  4 00:30:24 2026  580.126.09
# [0] NVIDIA H100 80GB HBM3 | 52°C,  55 % | 54984 / 81559 MB | exouser(54908M)
# [1] NVIDIA H100 80GB HBM3 | 57°C,  98 % | 54984 / 81559 MB | exouser(54908M)
# [2] NVIDIA H100 80GB HBM3 | 59°C,  95 % | 54984 / 81559 MB | exouser(54908M)
# [3] NVIDIA H100 80GB HBM3 | 56°C,  98 % | 54984 / 81559 MB | exouser(54908M)

## Training Log from Current Configuration ##
# Total time = 180 s. * 300 epochs ~ 14.5 Hours
# [Epoch 001] Time: 206.6541s | [Train] Loss: 7.03573551 Accuracy: Top1: 0.0000%, Top5: 0.0000% | [Test] Loss: 6.97415876 Accuracy: Top1: 0.0939%, Top5: 0.5186%
# [Epoch 002] Time: 166.2754s | [Train] Loss: 6.81198255 Accuracy: Top1: 0.0000%, Top5: 0.0000% | [Test] Loss: 6.19230747 Accuracy: Top1: 2.2541%, Top5: 7.4609%
