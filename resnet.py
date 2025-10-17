import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from types import SimpleNamespace

from activation import GELU_a, SiLU_a, ZiLU

class ResNet(nn.Module):
    def __init__(self, args):
        super(ResNet, self).__init__()
        self.args = args 
        self.num_classes = args.num_classes
        """
        ResNet-18 Params: 11,689,512
        ResNet-34 Params: 21,797,672
        ResNet-50 Params: 25,557,032
        """
        
        self.name = f"{args.model} - {args.activation}"

        self.activation = args.activation

        # Activation selection
        if self.activation == "zilu": 
            self.activation1 = ZiLU(s=args.s, inplace=args.inplace)
        if self.activation == "silu_a": 
            self.activation1 = SiLU_a(a=args.a, inplace=args.inplace)
        if self.activation == "gelu_a": 
            self.activation1 = GELU_a(a=args.a, inplace=args.inplace)
        if self.activation == "relu": 
            self.activation1 = nn.ReLU(inplace=args.inplace)
        if self.activation == "silu": 
            self.activation1 = nn.SiLU(inplace=args.inplace)
        if self.activation == "gelu": 
            self.activation1 = nn.GELU()

        self.first_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64), 
            self.activation1,
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Resnet 18 - [64, 128, 256, 512] * [2, 2, 2, 2]
        # Resnet 34 - [64, 128, 256, 512] * [3, 4, 6, 3]
        # Resnet 50 - [64, 128, 256, 512] * [3, 4, 6, 3]

        if args.model == "resnet18":
            layers = [2, 2, 2, 2]
        elif args.model == "resnet34":
            layers = [3, 4, 6, 3]
        elif args.model == "resnet50":
            layers = [0, 0, 0, 0] # Placeholder for ResNet-50, needs Bottleneck implementation
        else:
            raise ValueError("Invalid model type. Choose from 'resnet18', 'resnet34', 'resnet50'")

        self.in_channels = 64

        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1])
        self.layer3 = self._make_layer(256, layers[2])
        self.layer4 = self._make_layer(512, layers[3])


        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(start_dim=1),
            nn.Linear(512 * 7 * 7, self.num_classes)
        )

        

    def _make_layer(self, out_channels, blocks):
        layers = []

        # First block 
        layers.append(ResBlock(self.args, self.in_channels, out_channels))
        self.in_channels = out_channels 

        # Remaining blocks 
        for _ in range(1, blocks):
            layers.append(ResBlock(self.args, self.in_channels, out_channels))

        return nn.Sequential(*layers)
    
    def parameter_count(self): 
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params

    def forward(self, x):
        x = self.first_conv(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.classifier(x)

        return x
        


class ResBlock(nn.Module):
    def __init__(self, 
                 args, 
                 in_channels, 
                 out_channels, 
                 ):

        super(ResBlock, self).__init__()
        self.args = args
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.activation = args.activation

        # Activation selection
        if self.activation == "zilu": 
            self.activation1 = ZiLU(s=args.s, inplace=args.inplace)
            self.activation2 = ZiLU(s=args.s, inplace=args.inplace)
            self.activation3 = ZiLU(s=args.s, inplace=args.inplace)
        if self.activation == "silu_a": 
            self.activation1 = SiLU_a(a=args.a, inplace=args.inplace)
            self.activation2 = SiLU_a(a=args.a, inplace=args.inplace)
            self.activation3 = SiLU_a(a=args.a, inplace=args.inplace)
        if self.activation == "gelu_a": 
            self.activation1 = GELU_a(a=args.a, inplace=args.inplace)
            self.activation2 = GELU_a(a=args.a, inplace=args.inplace)
            self.activation3 = GELU_a(a=args.a, inplace=args.inplace)
        if self.activation == "relu": 
            self.activation1 = nn.ReLU(inplace=args.inplace)
            self.activation2 = nn.ReLU(inplace=args.inplace)
            self.activation3 = nn.ReLU(inplace=args.inplace)
        if self.activation == "silu": 
            self.activation1 = nn.SiLU(inplace=args.inplace)
            self.activation2 = nn.SiLU(inplace=args.inplace)
            self.activation3 = nn.SiLU(inplace=args.inplace)
        if self.activation == "gelu": 
            self.activation1 = nn.GELU()
            self.activation2 = nn.GELU()
            self.activation3 = nn.GELU()


        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            self.activation1
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            self.activation2
        )
        
        # Identity mapping
        if in_channels != out_channels:
            self.identity = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        else:
            self.identity = nn.Identity()

        self.activation = self.activation3 

    def forward(self, x):
        identity = self.identity(x)

        out = self.layer1(x)
        out = self.layer2(out)

        out += identity
        out = self.activation(out)

        return out
