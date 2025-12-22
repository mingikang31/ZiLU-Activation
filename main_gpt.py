"""Main File for ZiLU Activation Function for Training GPT2 Models"""

import argparse 
from pathlib import Path 
import os 
import torch 

# Models 
from Models.gpt2 import GPT2

from utils import write_to_file, set_seed 

def args_parser():
    parser = argparse.ArgumentParser(description="ZiLU Activation Function Experiments for GPT2 Models")

    # Model Args
    parser.add_argument('--vocab_size', type=int, default=50257, help='Vocabulary size for GPT2 model')
    parser.add_argument('--max_seq_length', type=int, default=1024, help='Maximum sequence length for GPT2 model')
    parser.add_argument('--embedding_dim', type=int, default=768, help='Embedding dimension for GPT2 model')
    parser.add_argument('--num_attention_heads', type=int, default=12, help='Number of attention heads for GPT2 model')
    parser.add_argument('--num_layers', type=int, default=12, help='Number of transformer layers for GPT2 model')

    # Activation Function Args 
    parser.add_argument('--activation', type=str, default='zilu', choices=['relu', 'gelu', 'silu', 'gelu_a', 'silu_a', 'zilu'], help='Activation function to use')
    parser.add_argument('--a', type=float, default=1.0, help='Parameter a for gelu_a and silu_a')
    parser.add_argument('--s', type=float, default=1.0, help='Parameter s for zilu')
    parser.add_argument('--inplace', action='store_true', help='Use inplace activation functions')
    parser.set_defaults(inplace=False)

    # Arguments for Data 
    parser.add_argument("--dataset", type=str, default="wikitext103", choices=["wikitext103"], help="Dataset to use for training and evaluation")
    parser.add_argument("--data_path", type=str, default="./Data", help="Path to the dataset")

    # Training Arguments
    parser.add_argument("--batch_size", type=int, default=12, help="Batch size for training and evaluation")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs for training")
    parser.add_argument("--use_amp", action="store_true", help="Use mixed precision training")
    parser.set_defaults(use_amp=False)
    parser.add_argument("--clip_grad_norm", type=float, default=1.0, help="Gradient clipping value")

    # Optimizer Arguments
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'adamw'], help='Default Optimizer: adamw')
    parser.add_argument('--weight_decay', type=float, default=5e-2, help='Weight decay for optimizer') # For Adam & Adamw

    # Learning Rate Arguments 
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate for the optimizer")
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['step', 'cosine', 'plateau'], help='Learning rate scheduler type')
    parser.add_argument('--lr_step', type=int, default=20, help='Step size for learning rate scheduler') # Only for StepLR
    parser.add_argument('--lr_gamma', type=float, default=0.5, help='Gamma for StepLR scheduler') # Only for StepLR

    # Device Arguments 
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use for training and evaluation')
    parser.add_argument('--seed', default=0, type=int)

    # Output Arguments 
    parser.add_argument("--output_dir", type=str, default="./Output/ZiLU/GPT2", help="Directory to save the output files")

    # Test Arguments
    parser.add_argument("--test_only", action="store_true", help="Only test the model")
    parser.set_defaults(test_only=False)

    return parser 

def main(args):
    # Set Seed
    if args.seed != 0:
        set_seed(args.seed)

    # Create Output Directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize Model
    model = GPT2(
        vocab_size=args.vocab_size,
        max_seq_length=args.max_seq_length,
        embedding_dim=args.embedding_dim,
        num_attention_heads=args.num_attention_heads,
        num_layers=args.num_layers,
        device=args.device
    )

    # Print Model Summary
    model_summary = str(model)
    print(model_summary)

    # Training and Evaluation Logic Here
    # ...

    if args.test_only: 
        ex = torch.randint(0, args.vocab_size, (1, args.max_seq_length)).to(args.device)
        out = model(ex) 
        print("Test output shape:", out.shape)
    else: 
        pass 