"""Main File for ZiLU Activation Function"""


import argparse 
from pathlib import Path 
import os 

import torch 

# Datasets and Eval 
from train_eval import Train_Eval
from dataset import CIFAR10, CIFAR100, ImageNet

# Models 
from Models.vgg import VGG
from Models.resnet import ResNet
from Models.vit import ViT


# Utils 
from utils import write_to_file, set_seed

# Activation Functions


def args_parser():
    parser = argparse.ArgumentParser(description="Activation Function Experiments")

    # Model Args 
    parser.add_argument('--activation', type=str, default='zilu', choices=['relu', 'gelu', 'silu', 'gelu_a', 'silu_a', 'zilu'], help='Activation function to use')
    parser.add_argument('--a', type=float, default=1.0, help='Parameter a for gelu_a and silu_a')
    parser.add_argument('--s', type=float, default=1.0, help='Parameter s for zilu')
    parser.add_argument('--inplace', action='store_true', help='Use inplace activation functions')
    parser.set_defaults(inplace=False)

    parser.add_argument('--model', type=str, default='vgg11', choices=['vgg11', 'vgg13', 'vgg16', 'vgg19', 'resnet18', 'resnet50', 'vit-tiny'], help='Model architecture')

    # Arguments for Data 
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100", 'imagenet'], help="Dataset to use for training and evaluation")
    parser.add_argument("--data_path", type=str, default="./Data", help="Path to the dataset")
        
    # Training Arguments
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training and evaluation")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs for training")
    parser.add_argument("--use_amp", action="store_true", help="Use mixed precision training")
    parser.set_defaults(use_amp=False)
    parser.add_argument("--clip_grad_norm", type=float, default=1.0, help="Gradient clipping value")
    
    
    # Loss Function Arguments
    parser.add_argument("--criterion", type=str, default="CrossEntropy", choices=["CrossEntropy", "MSE"], help="Loss function to use for training")
    
    # Optimizer Arguments 
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'sgd', 'adamw'], help='Default Optimizer: adamw')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD optimizer') # Only for SGD
    parser.add_argument('--weight_decay', type=float, default=5e-2, help='Weight decay for optimizer') # For Adam & Adamw
    
    # Learning Rate Arguments
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate for the optimizer")
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
    
    return parser



def main(args):

    # Resize False 
    args.resize = False
    
    # Dataset 
    if args.dataset == "cifar10":
        dataset = CIFAR10(args)
        args.num_classes = dataset.num_classes 
        args.img_size = dataset.img_size 
    elif args.dataset == "cifar100":
        dataset = CIFAR100(args)
        args.num_classes = dataset.num_classes 
        args.img_size = dataset.img_size 
    elif args.dataset == "imagenet":
        dataset = ImageNet(args)
        args.num_classes = dataset.num_classes 
        args.img_size = dataset.img_size
    else:
        raise ValueError("Dataset not supported")

    # VGG Models 
    if args.model == "vgg11":
        features_config = "A"
        model = VGG(args, features_config=features_config, dropout=0.5)
    elif args.model == "vgg13":
        features_config = "B"
        model = VGG(args, features_config=features_config, dropout=0.5)
    elif args.model == "vgg16":
        features_config = "D"
        model = VGG(args, features_config=features_config, dropout=0.5)
    elif args.model == "vgg19":
        features_config = "E"
        model = VGG(args, features_config=features_config, dropout=0.5)

    # ResNet
    if args.model == "resnet18" or args.model == "resnet50":
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

    model.to(args.device)
    print(f"Model: {model.name}")

    # Parameters
    total_params, trainable_params = model.parameter_count()
    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params}")
    args.total_params = total_params
    args.trainable_params = trainable_params

    
    if args.test_only:
        ex = torch.Tensor(3, 3, 32, 32).to(args.device)
        out = model(ex)
        print(f"Output shape: {out.shape}")
        print("Testing Complete")
    else:
        # Check if the output directory exists, if not create it
        if args.output_dir:
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)

        # Set the seed for reproducibility
        set_seed(args.seed)
        
        # Training Modules 
        train_eval_results = Train_Eval(args, 
                                    model, 
                                    dataset.train_loader, 
                                    dataset.test_loader
                                    )
        
        # Storing Results in output directory 
        write_to_file(os.path.join(args.output_dir, "args.txt"), args)
        write_to_file(os.path.join(args.output_dir, "model.txt"), model)
        write_to_file(os.path.join(args.output_dir, "train_eval_results.txt"), train_eval_results)

if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description="Activation Functions", parents=[args_parser()], add_help=False)
    args = parser.parse_args()

    main(args)
