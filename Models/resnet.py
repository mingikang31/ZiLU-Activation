import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from types import SimpleNamespace

# Activation Functions 
from Models.activation import (GELU_s, SiLU_s, ZiLU_Old, ArcTan,
                               ArcTan_Approx, ZiLU, ZiLU_Approx, SquarePlus)

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

        # Activation Selection
        self.activation = args.activation

        # Activation function mapping
        self.activation_map = {
            "relu": lambda: nn.ReLU(inplace=args.inplace), 
            "silu": lambda: nn.SiLU(inplace=args.inplace), 
            "gelu": lambda: nn.GELU(), 
            "sigmoid": lambda: nn.Sigmoid(), 

            # Previous Activation Generation
            "gelu_s": lambda: GELU_s(sigma=args.sigma, inplace=args.inplace), 
            "silu_s": lambda: SiLU_s(sigma=args.sigma, inplace=args.inplace), 
            "zilu_old": lambda: ZiLU_Old(sigma=args.sigma, inplace=args.inplace), 

            # Current Activation Generation 
            "arctan": lambda: ArcTan(sigma=args.sigma), 
            "arctan_approx": lambda: ArcTan_Approx(sigma=args.sigma), 
            "zilu": lambda: ZiLU(sigma=args.sigma), 
            "zilu_approx": lambda: ZiLU_Approx(sigma=args.sigma), 
            "squareplus": lambda: SquarePlus(beta=4),

            # Other Activations
            "leaky_relu": lambda: nn.LeakyReLU(inplace=args.inplace), 
            "prelu": lambda: nn.PReLU(), 
            "elu": lambda: nn.ELU(inplace=args.inplace), 
            "hardshrink": lambda: nn.Hardshrink(), 
            "softshrink": lambda: nn.Softshrink(), 
            "tanhshrink": lambda: nn.Tanhshrink(), 
            "softplus": lambda: nn.Softplus(),
            "softsign": lambda: nn.Softsign(), 
            "tanh": lambda: nn.Tanh(),
            "celu": lambda: nn.CELU(inplace=args.inplace),
            "mish": lambda: nn.Mish(inplace=args.inplace), 
            "hardswish": lambda: nn.Hardswish(inplace=args.inplace), 
            "hardsigmoid": lambda: nn.Hardsigmoid(inplace=args.inplace),
            "selu": lambda: nn.SELU(inplace=args.inplace),
            "hardtanh": lambda: nn.Hardtanh(inplace=args.inplace)
        }

        if self.activation not in self.activation_map:
            raise ValueError(f"Unsupported activation function: {self.activation}")
            
        self.activation_first_conv = self.activation_map[self.activation]()

        self.first_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64), 
            self.activation_first_conv,
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Resnet 18 - [64, 128, 256, 512] * [2, 2, 2, 2]
        # Resnet 34 - [64, 128, 256, 512] * [3, 4, 6, 3]
        # Resnet 50 - [64, 128, 256, 512] * [3, 4, 6, 3]

        if args.model == "resnet18":
            layers = [2, 2, 2, 2]
            block = ResBlock 
            self.expansion = 1
        elif args.model == "resnet34":
            layers = [3, 4, 6, 3]
            block = ResBlock
            self.expansion = 1
        elif args.model == "resnet50":
            layers = [3, 4, 6, 3]
            block = BottleNeck
            self.expansion = 4
        else:
            raise ValueError("Invalid model type. Choose from 'resnet18', 'resnet34', 'resnet50'")

        self.in_channels = 64

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)


        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=1),
            nn.Linear(512 * self.expansion, self.num_classes)
        )

    def _make_layer(self, block, out_channels, blocks, stride=1):
        layers = []

        # First block handles stride and channel change
        layers.append(block(self.args, self.in_channels, out_channels, stride=stride))

        if block == BottleNeck: 
            self.in_channels = out_channels * block.expansion
        else: 
            self.in_channels = out_channels
        
        # Remaining blocks use stride = 1
        for _ in range(1, blocks):
            layers.append(block(self.args, self.in_channels, out_channels, stride=1))

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
                 stride=1
                 ):

        super(ResBlock, self).__init__()
        self.args = args
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride 

        # Activation Selection
        self.activation = args.activation

        # Activation function mapping
        self.activation_map = {
            "relu": lambda: nn.ReLU(inplace=args.inplace), 
            "silu": lambda: nn.SiLU(inplace=args.inplace), 
            "gelu": lambda: nn.GELU(), 
            "sigmoid": lambda: nn.Sigmoid(), 

            # Previous Activation Generation
            "gelu_s": lambda: GELU_s(sigma=args.sigma, inplace=args.inplace), 
            "silu_s": lambda: SiLU_s(sigma=args.sigma, inplace=args.inplace), 
            "zilu_old": lambda: ZiLU_Old(sigma=args.sigma, inplace=args.inplace), 

            # Current Activation Generation 
            "arctan": lambda: ArcTan(sigma=args.sigma), 
            "arctan_approx": lambda: ArcTan_Approx(sigma=args.sigma), 
            "zilu": lambda: ZiLU(sigma=args.sigma), 
            "zilu_approx": lambda: ZiLU_Approx(sigma=args.sigma), 
            "squareplus": lambda: SquarePlus(beta=4),

            # Other Activations
            "leaky_relu": lambda: nn.LeakyReLU(inplace=args.inplace), 
            "prelu": lambda: nn.PReLU(), 
            "elu": lambda: nn.ELU(inplace=args.inplace), 
            "hardshrink": lambda: nn.Hardshrink(), 
            "softshrink": lambda: nn.Softshrink(), 
            "tanhshrink": lambda: nn.Tanhshrink(), 
            "softplus": lambda: nn.Softplus(),
            "softsign": lambda: nn.Softsign(), 
            "tanh": lambda: nn.Tanh(),
            "celu": lambda: nn.CELU(inplace=args.inplace),
            "mish": lambda: nn.Mish(inplace=args.inplace), 
            "hardswish": lambda: nn.Hardswish(inplace=args.inplace), 
            "hardsigmoid": lambda: nn.Hardsigmoid(inplace=args.inplace),
            "selu": lambda: nn.SELU(inplace=args.inplace),
            "hardtanh": lambda: nn.Hardtanh(inplace=args.inplace)            
        }

        if self.activation not in self.activation_map:
            raise ValueError(f"Unsupported activation function: {self.activation}")

        self.activation1 = self.activation_map[self.activation]()
        self.final_activation = self.activation_map[self.activation]() 


        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            self.activation1
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        
        # Identity mapping
        if stride != 1 or in_channels != out_channels: 
            self.identity = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False), 
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.identity = nn.Identity()


    def forward(self, x):
        identity = self.identity(x)

        out = self.layer1(x)
        out = self.layer2(out)

        out += identity
        out = self.final_activation(out)

        return out

class BottleNeck(nn.Module):
    expansion = 4
    
    def __init__(self, 
                 args, 
                 in_channels, 
                 out_channels, 
                 stride=1
                 ):
        super(BottleNeck, self).__init__()
        self.args = args
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        
        # Activation Selection
        self.activation = args.activation

        # Activation function mapping
        self.activation_map = {
            "relu": lambda: nn.ReLU(inplace=args.inplace), 
            "silu": lambda: nn.SiLU(inplace=args.inplace), 
            "gelu": lambda: nn.GELU(), 
            "sigmoid": lambda: nn.Sigmoid(), 

            # Previous Activation Generation
            "gelu_s": lambda: GELU_s(sigma=args.sigma, inplace=args.inplace), 
            "silu_s": lambda: SiLU_s(sigma=args.sigma, inplace=args.inplace), 
            "zilu_old": lambda: ZiLU_Old(sigma=args.sigma, inplace=args.inplace), 

            # Current Activation Generation 
            "arctan": lambda: ArcTan(sigma=args.sigma), 
            "arctan_approx": lambda: ArcTan_Approx(sigma=args.sigma), 
            "zilu": lambda: ZiLU(sigma=args.sigma), 
            "zilu_approx": lambda: ZiLU_Approx(sigma=args.sigma), 
            "squareplus": lambda: SquarePlus(beta=4),

            # Other Activations
            "leaky_relu": lambda: nn.LeakyReLU(inplace=args.inplace), 
            "prelu": lambda: nn.PReLU(), 
            "elu": lambda: nn.ELU(inplace=args.inplace), 
            "hardshrink": lambda: nn.Hardshrink(), 
            "softshrink": lambda: nn.Softshrink(), 
            "tanhshrink": lambda: nn.Tanhshrink(), 
            "softplus": lambda: nn.Softplus(),
            "softsign": lambda: nn.Softsign(), 
            "tanh": lambda: nn.Tanh(),
            "celu": lambda: nn.CELU(inplace=args.inplace),
            "mish": lambda: nn.Mish(inplace=args.inplace), 
            "hardswish": lambda: nn.Hardswish(inplace=args.inplace), 
            "hardsigmoid": lambda: nn.Hardsigmoid(inplace=args.inplace),
            "selu": lambda: nn.SELU(inplace=args.inplace),
            "hardtanh": lambda: nn.Hardtanh(inplace=args.inplace)
        }

        if self.activation not in self.activation_map:
            raise ValueError(f"Unsupported activation function: {self.activation}")

        self.activation1 = self.activation_map[self.activation]()
        self.activation2 = self.activation_map[self.activation]()
        self.final_activation = self.activation_map[self.activation]() 

        # 1x1 conv to reduce channels 
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1), 
            nn.BatchNorm2d(out_channels),
            self.activation1
        )

        # 3x3 conv with stride (main processing) 
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            self.activation2
        )

        # 1x1 conv to expand channels 
        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels * self.expansion)
        )

        # Identity Mapping 
        if stride != 1 or in_channels != out_channels * self.expansion: 
            self.identity = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride), 
                nn.BatchNorm2d(out_channels * self.expansion)
            )
        else: 
            self.identity = nn.Identity() 

    def forward(self, x):
        identity = self.identity(x) 

        out = self.conv1(x) 
        out = self.conv2(out) 
        out = self.conv3(out)

        out += identity 
        out = self.final_activation(out)

        return out