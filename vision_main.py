"""Main File for ZiLU Activation Function"""
import argparse 
from pathlib import Path 
import os 
import torch 
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# Datasets and Eval 
from train_eval import Train_Eval, Train_Eval_ImageNet, setup_distributed, cleanup_distributed
from dataset import CIFAR10, CIFAR100, ImageNet1K, mixup_fn

# Models 
from Models.vgg import VGG
from Models.resnet import ResNet
from Models.vit import ViT

# Utils 
from utils import write_to_file, set_seed

"""
Dropout for ViT/Swin Transformer:
    - Swin-T = 0.2
    - Swin-S = 0.3
    - Swin-B = 0.5
    
"""


def args_parser():
    parser = argparse.ArgumentParser(description="Activation Function Experiments")

    # Model Args
    parser.add_argument('--activation', type=str, default='relu', choices=[
        'relu', 'gelu', 'silu', 'sigmoid', 'gelu_s', 'silu_s', 'zilu_old', 
        'arctan', 'arctan_approx', 'zilu', 'zilu_approx', 'leaky_relu', 'prelu', 
        'elu', 'hardshrink', 'softshrink', 'tanhshrink', 'hardtanh', 'softplus', 'softsign', 
        'tanh', 'celu', 'mish', 'hardswish', 'hardsigmoid', 'selu', 'squareplus'
    ], help='Activation function to use')
    parser.add_argument('--sigma', type=float, default=None, help='Sigma parameter for ZiLU activation function')
    parser.add_argument('--inplace', action='store_true', help='Use inplace activation functions')
    parser.set_defaults(inplace=False)

    parser.add_argument('--model', type=str, default='vgg11', choices=['vgg11', 'vgg13', 'vgg16', 'vgg19', 'resnet18', 'resnet34', 'resnet50', 'vit-tiny', 'vit-small', 'vit-medium', 'vit-large'], help='Model architecture')

    # Arguments for Data 
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100", "imagenet1k"], help="Dataset to use for training and evaluation")
    parser.add_argument("--resize", type=int, default=None, help="Resize images to 224x224")
    parser.add_argument("--augment", action="store_true", help="Use data augmentation")
    parser.set_defaults(augment=False)
    parser.add_argument("--noise", type=float, default=0.0, help="Standard deviation of Gaussian noise to add to the data")
    parser.add_argument("--data_path", type=str, default="/mnt/research/j.farias/mkang2/Datasets", help="Path to the dataset")

        
    # Training Arguments
    parser.add_argument("--compile", action="store_true", help="Use compiled model for training and evaluation")
    parser.set_defaults(compile=False)
    parser.add_argument("--compile_mode", type=str, default="default", choices=["default", "reduce-overhead", "reduce-memory", "reduce-overhead", "max-autotune"], help="Compilation mode for torch.compile")
    
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training and evaluation")
    parser.add_argument("--num_epochs", type=int, default=150, help="Number of epochs for training")
    parser.add_argument("--use_amp", action="store_true", help="Use mixed precision training")
    parser.set_defaults(use_amp=False)
    parser.add_argument("--clip_grad_norm", type=float, default=1.0, help="Gradient clipping value")

    # Data Loader Specific
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--persistent_workers", action="store_true", help="Use persistent workers for data loading")
    parser.set_defaults(persistent_workers=False)
    parser.add_argument("--prefetch_factor", type=int, default=None, help="Prefetch factor for data loading")
    parser.add_argument("--pin_memory", action="store_true", help="Pin memory for data loading")
    parser.set_defaults(pin_memory=False)
    
    # Loss Function Arguments
    parser.add_argument("--criterion", type=str, default="CrossEntropy", choices=["CrossEntropy", "MSE"], help="Loss function to use for training")
    
    # Optimizer Arguments 
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'sgd', 'adamw'], help='Default Optimizer: adamw')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD optimizer') # Only for SGD
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='Weight decay for optimizer') # For Adam & Adamw
    
    # Learning Rate Arguments
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for the optimizer")
    parser.add_argument('--lr_step', type=int, default=20, help='Step size for learning rate scheduler') # Only for StepLR
    parser.add_argument('--lr_gamma', type=float, default=0.1, help='Gamma for learning rate scheduler') # Only for StepLR
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['step', 'cosine', 'plateau'], help='Learning rate scheduler')
    
    # Device Arguments
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda", "mps"], help="Device to use for training and evaluation")
    parser.add_argument('--seed', default=0, type=int)

    # Output Arguments 
    parser.add_argument("--output_dir", type=str, default="./Output/ZiLU/vgg11", help="Directory to save the output files")
    
    # Test Arguments
    parser.add_argument("--test_only", action="store_true", help="Only test the model")
    parser.set_defaults(test_only=False)

    # Distributed Data Parallel (DDP) 
    parser.add_argument("--ddp", action="store_true", help="Use Distributed Data Parallel (DDP) for training")
    parser.set_defaults(ddp=False)
    parser.add_argument("--ddp_batch_size", type=int, default=128, help="Batch size per GPU for DDP training")
    
    return parser

def main(args):
    # Using TensorFloat-32 (TF32)
    try: 
        torch.set_float32_matmul_precision('high')
    except:
        print("Could not use TensorFloat-32")

    # DDP Setup 
    
    local_rank = setup_distributed()
    if dist.is_initialized() and dist.is_available():
        args.ddp = True 
        print(f"DDP is initialized. Local Rank: {local_rank}, World Size: {dist.get_world_size()}")
    
    # Dataset 
    if args.dataset == "cifar10":
        dataset = CIFAR10(args)
        args.num_classes = dataset.num_classes 
        args.img_size = dataset.img_size 
    elif args.dataset == "cifar100":
        dataset = CIFAR100(args)
        args.num_classes = dataset.num_classes 
        args.img_size = dataset.img_size 
    elif args.dataset == "imagenet1k":
        args.batch_size = 1024 # Standard Batch Size for ImageNet-1K
        args.augment = True
        dataset = ImageNet1K(args)
        args.num_classes = dataset.num_classes 
        args.img_size = dataset.img_size
    else:
        raise ValueError("Dataset not supported")

    # VGG Models 
    if args.model == "vgg11":
        model = VGG(args, features_config="A", dropout=0.5)
    elif args.model == "vgg13":
        model = VGG(args, features_config="B", dropout=0.5)
    elif args.model == "vgg16":
        model = VGG(args, features_config="D", dropout=0.5)
    elif args.model == "vgg19":
        model = VGG(args, features_config="E", dropout=0.5)

    # ResNet
    if args.model in ["resnet18", "resnet34", "resnet50"]:
        model = ResNet(args)        

    # ViT 
    if args.model == "vit-tiny":
        args.patch_size = 16
        args.d_hidden = 192
        args.d_mlp = 768
        args.num_heads = 3
        args.num_layers = 12
        args.dropout = 0.1
        args.attention_dropout = 0.1
        model = ViT(args)
        
    elif args.model == "vit-small":
        args.patch_size = 16 
        args.d_hidden = 384
        args.d_mlp = 1536
        args.num_heads = 6 
        args.num_layers = 12 
        args.dropout = 0.1
        args.attention_dropout = 0.1
        model = ViT(args)

    elif args.model == "vit-medium": 
        args.patch_size = 16 
        args.d_hidden = 512 
        args.d_mlp = 2048 
        args.num_heads = 8
        args.num_layers = 12
        args.dropout = 0.1 
        args.attention_dropout = 0.1
        model = ViT(args) 

    elif args.model == "vit-large": 
        args.patch_size = 16 
        args.d_hidden = 768
        args.d_mlp = 3072 
        args.num_heads = 12 
        args.num_layers = 12 
        args.dropout = 0.1 
        args.attention_dropout = 0.1
        model = ViT(args)

    model.to(args.device)
    print(f"Model: {model.name}")

    # Distributed Data Parallel
    if args.ddp and dist.is_available() and dist.is_initialized():
        model = DDP(model, device_ids=[local_rank])

    # Parameters
    total_params, trainable_params = model.parameter_count()
    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params}")
    args.total_params = total_params
    args.trainable_params = trainable_params

    
    if args.test_only:
        ex = torch.Tensor(3, args.img_size[0], args.img_size[1], args.img_size[2]).to(args.device)
        out = model(ex)
        print(f"Output shape: {out.shape}")
        print("Testing Complete")
    else:
        # Check if the output directory exists, if not create it
        if args.output_dir:
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)

        # Training Module
        if args.dataset in ["cifar10", "cifar100"]:
            train_eval_results = Train_Eval(args, 
                                        model, 
                                        dataset.train_loader, 
                                        dataset.test_loader
                                        )
        elif args.dataset == "imagenet1k":
            train_eval_results = Train_Eval_ImageNet(
                                        args, 
                                        model, 
                                        dataset.train_loader, 
                                        dataset.test_loader,
                                        train_sampler=dataset.train_sampler,
                                        mixup_fn=mixup_fn, 
                                        rank=local_rank
                                        )

        # Cleanup DDP
        cleanup_distributed()
        
        # Store Results
        write_to_file(os.path.join(args.output_dir, "args.txt"), args)
        write_to_file(os.path.join(args.output_dir, "model.txt"), model)
        write_to_file(os.path.join(args.output_dir, "train_eval_results.txt"), train_eval_results)

if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description="Activation Functions", parents=[args_parser()], add_help=False)
    args = parser.parse_args()
    main(args)
