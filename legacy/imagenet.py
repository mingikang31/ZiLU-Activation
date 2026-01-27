import os 
import argparse 
import math 
import torch 
import torch.nn as nn 
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models 
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

# Timm for Vision Transformers
from timm.data import create_transform 
from timm.loss import SoftTargetCrossEntropy
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.data.mixup import Mixup

configuration = {
    "epochs": 300, 
    "batch_size": 1024,
    "base_lr": 5e-4, 
    "weight_decay": 0.05,
    "input_size": 224,
    "num_classes": 1000, 
    "mixup_alpha": 0.8, # Mixup strength
    "cutmix_alpha": 1.0, # CutMix strength 
    "mixup_prob": 1.0,
    "label_smoothing": 0.1,
    "num_workers": 8, 
    "print_freq": 100,
    
}


'''ImageNet1K Dataset Class'''
class ImageNet1K:
    def __init__(self, args):
        self.img_size = (3, args.resize, args.resize)
        self.num_classes = 1000

        # Transformations
        train_transform = transforms.Compose([
            transforms.Resize((args.resize, args.resize)),
            transforms.RandomResizedCrop(args.resize),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        test_transform = transforms.Compose([
            transforms.Resize((args.resize, args.resize)),
            transforms.CenterCrop(args.resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Load Datasets
        self.train_data = datasets.ImageNet(root=args.data_path, split='train', transform=train_transform)
        self.test_data = datasets.ImageNet(root=args.data_path, split='val', transform=test_transform)

        # Data Loaders
        self.train_loader = DataLoader(
            dataset=self.train_data,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            persistent_workers=args.persistent_workers,
            prefetch_factor=args.prefetch_factor,
            pin_memory=args.pin_memory)

        self.test_loader = DataLoader(
            dataset=self.test_data,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            persistent_workers=args.persistent_workers,
            prefetch_factor=args.prefetch_factor,
            pin_memory=args.pin_memory)

