
import torch 
import torch.nn as nn 
import torch.nn.functional as F

from activation import GELU_a, SiLU_a, ZiLU


r"""
@software{torchvision2016,
    title        = {TorchVision: PyTorch's Computer Vision library},
    author       = {TorchVision maintainers and contributors},
    year         = 2016,
    journal      = {GitHub repository},
    publisher    = {GitHub},
    howpublished = {\url{https://github.com/pytorch/vision}}
}
"""

class VGG(nn.Module):
    def __init__(
        self, 
        args,
        # in_channels=3, 
        features_config="A", 
        # num_classes=1000,
        dropout=0.5    
    ):
        super(VGG, self).__init__()
        """
        A: VGG-11 Params: 132,868,840
        B: VGG-13 Params: 133,053,736
        D: VGG-16 Params: 138,365,992
        E: VGG-19 Params: 143,678,248
        features_config: str, one of "A", "B", "D", "E"
        """

        self.args = args 
        in_channels = self.args.img_size[0] 
        num_classes = self.args.num_classes

        self.name = f"VGG {features_config} {args.activation}"

        self.activation = args.activation 
        

        cfg = {
            "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
            "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
            "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
            "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
        }

        layers = [] 

        for v in cfg[features_config]:
            if v == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=3, stride=1, padding=1), 
                         nn.BatchNorm2d(v)]

                if self.activation == "zilu": 
                    layers += [ZiLU(s=args.s, inplace=args.inplace)]
                if self.activation == "silu_a": 
                    layers += [SiLU_a(a=args.a, inplace=args.inplace)]
                if self.activation == "gelu_a": 
                    layers += [GELU_a(a=args.a, inplace=args.inplace)]
                if self.activation == "relu": 
                    layers += [nn.ReLU(inplace=args.inplace)]
                if self.activation == "silu": 
                    layers += [nn.SiLU(inplace=args.inplace)]
                if self.activation == "gelu": 
                    layers += [nn.GELU()]
                        
                in_channels = v

        self.features = nn.Sequential(*layers)

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(4096, num_classes)
        )

        self.to(self.args.device)


    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def parameter_count(self): 
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params




    