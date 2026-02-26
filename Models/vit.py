"""VIT Model"""

import torch
import torch.nn as nn
import torch.nn.functional as F 
from torchsummary import summary 
import numpy as np


# Activation Functions 
from Models.activation import (GELU_s, SiLU_s, ZiLU_Old, ArcTan,
                               ArcTan_Approx, ZiLU, ZiLU_Approx, SquarePlus)

'''VGG Model Class'''
class ViT(nn.Module): 
    def __init__(self, args): 
        super(ViT, self).__init__()
        assert args.img_size[1] % args.patch_size == 0 and args.img_size[2] % args.patch_size == 0, "img_size dimensions must be divisible by patch_size dimensions"
        assert args.d_hidden % args.num_heads == 0, "d_hidden must be divisible by n_heads"
        
        self.args = args
        self.model = "VIT"
        
        self.d_hidden = self.args.d_hidden 
        self.d_mlp = self.args.d_mlp
        
        self.img_size = self.args.img_size[1:]
        self.n_classes = self.args.num_classes # Number of Classes
        self.n_heads = self.args.num_heads
        self.patch_size = (self.args.patch_size, self.args.patch_size) # Patch Size
        self.n_channels = self.args.img_size[0]
        self.n_layers = self.args.num_layers # Number of Layers
        
        self.n_patches = (self.img_size[0] * self.img_size[1]) // (self.patch_size[0] * self.patch_size[1])
        
        self.dropout = self.args.dropout # Dropout Rate
        self.attention_dropout = self.args.attention_dropout # Attention Dropout Rate   
        self.max_seq_length = self.n_patches + 1 # +1 for class token
        
        self.patch_embedding = PatchEmbedding(self.d_hidden, self.img_size, self.patch_size, self.n_channels) # Patch Embedding Layer
        self.positional_encoding = PositionalEncoding(self.d_hidden, self.max_seq_length)

        self.dpr = [x.item() for x in torch.linspace(0, self.args.drop_path_rate, self.n_layers)]  # stochastic depth decay rule

        # self.transformer_encoder = nn.Sequential(*[TransformerEncoder_DropPath(
        #     args=args, 
        #     d_hidden=self.d_hidden, 
        #     d_mlp=self.d_mlp, 
        #     num_heads=self.n_heads, 
        #     dropout=self.dropout, 
        #     attention_dropout=self.attention_dropout,
        #     drop_path=self.dpr[i]
        #     ) for i in range(self.n_layers)])
        
        self.transformer_encoder = nn.Sequential(*[TransformerEncoder(
            args=args, 
            d_hidden=self.d_hidden, 
            d_mlp=self.d_mlp, 
            num_heads=self.n_heads, 
            dropout=self.dropout, 
            attention_dropout=self.attention_dropout
            ) for _ in range(self.n_layers)])
        
        self.classifier = nn.Linear(self.d_hidden, self.n_classes)
        
        self.device = args.device
        
        self.to(self.device)
        self.name = f"{self.args.model} {self.args.activation}"
        
    def forward(self, x): 
        x = self.patch_embedding(x)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        x = self.classifier(x[:, 0]) # Taking the CLS token for classification
        return x

    def summary(self): 
        original_device = next(self.parameters()).device
        try:
            self.to("cpu")
            print(f"--- Summary for {self.name} ---")
            summary(self, input_size=self.img_size, device="cpu") 
        except Exception as e:
            print(f"Could not generate summary: {e}")
        finally:
            self.to(original_device)
        
    def parameter_count(self): 
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params
        
class PatchEmbedding(nn.Module): 
    def __init__(self, d_hidden, img_size, patch_size, n_channels=3): 
        super(PatchEmbedding, self).__init__()
        
        self.d_hidden = d_hidden # Dimensionality of Model 
        self.img_size = img_size # Size of Image
        self.patch_size = patch_size # Patch Size 
        self.n_channels = n_channels # Number of Channels in Image
        
        self.linear_projection = nn.Conv2d(in_channels=n_channels, out_channels=d_hidden, kernel_size=patch_size, stride=patch_size) # Linear Projection Layer
        # self.norm = nn.LayerNorm(d_hidden) # Normalization Layer
        self.norm = ViTBatchNorm(d_hidden)
        
        self.flatten = nn.Flatten(start_dim=2)
        
    def forward(self, x): 
        x = self.linear_projection(x) # (B, C, H, W) -> (B, d_hidden, H', W')
        x = self.flatten(x) # (B, d_hidden, H', W') -> (B, d_hidden, n_patches)
        x = x.transpose(1, 2) # (B, d_hidden, n_patches) -> (B, n_patches, d_hidden)
        x = self.norm(x) # (B, n_patches, d_hidden) -> (B, n_patches, d_hidden)
        return x
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_hidden, max_seq_length): 
        super(PositionalEncoding, self).__init__()
        
        self.cls_tokens = nn.Parameter(torch.randn(1, 1, d_hidden)) # Classification Token

        pe = torch.zeros(max_seq_length, d_hidden)  # Positional Encoding Tensor
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1) 
        div_term = torch.exp(torch.arange(0, d_hidden, 2).float() * (-np.log(10000.0) / d_hidden))  

        pe[:, 0::2] = torch.sin(position * div_term)  # Apply sine to even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Apply cosine to odd indices

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x): 
        # Expand to have class token for each image in batch 
        tokens_batch = self.cls_tokens.expand(x.shape[0], -1, -1) # (B, 1, d_hidden)
        
        # Concatenate class token with positional encoding
        x = torch.cat((tokens_batch, x), dim=1)
        
        # Add positional encoding to the input 
        x = x + self.pe[:, :x.size(1)].to(x.device) 
        return x

"""Multi-Head Layers for Transformer Encoder"""
class MultiHeadAttention(nn.Module): 
    def __init__(self, d_hidden, num_heads, attention_dropout):
        super(MultiHeadAttention, self).__init__()
        assert d_hidden % num_heads == 0, "d_hidden must be divisible by num_heads"
        
        self.d_hidden = d_hidden
        self.num_heads = num_heads
        self.d_k = d_hidden // num_heads # dimension of each head
        self.dropout = nn.Dropout(attention_dropout)
        
        self.W_q = nn.Linear(d_hidden, d_hidden, bias=False)
        self.W_k = nn.Linear(d_hidden, d_hidden, bias=False)
        self.W_v = nn.Linear(d_hidden, d_hidden, bias=False)
        self.W_o = nn.Linear(d_hidden, d_hidden) # Final Transformation before residual connection
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        attn_probs = self.dropout(torch.softmax(attn_scores, dim=-1))
        output = torch.matmul(attn_probs, V)
        return output, attn_probs
    
    def split_head(self, x): 
        batch_size, seq_length, d_hidden = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2) # (B, num_heads, seq_length, d_k)
        
    def combine_heads(self, x): 
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_hidden) 
    
    def forward(self, x, mask=None):
        q = self.split_head(self.W_q(x)) # (B, num_heads, seq_length, d_k)
        k = self.split_head(self.W_k(x))
        v = self.split_head(self.W_v(x))
        
        attn_output, _ = self.scaled_dot_product_attention(q, k, v, mask) # (B, num_heads, seq_length, d_k)
        output = self.W_o(self.combine_heads(attn_output)) # (B, seq_length, d_hidden)
        return output


class TransformerEncoder(nn.Module): 
    def __init__(self, args, d_hidden, d_mlp, num_heads, dropout, attention_dropout):
        super(TransformerEncoder, self).__init__()
        self.args = args 

        self.d_hidden = d_hidden 
        self.d_mlp = d_mlp
        self.num_heads = num_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout

        self.attention = MultiHeadAttention(d_hidden, num_heads, attention_dropout)

        # self.norm1 = nn.LayerNorm(d_hidden)
        # self.norm2 = nn.LayerNorm(d_hidden)

        # ViT Batch Norm 
        self.norm1 = ViTBatchNorm(d_hidden)
        self.norm2 = ViTBatchNorm(d_hidden)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

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
            "hardtanh": lambda: nn.Hardtanh(inplace=args.inplace), 
            "identity": lambda: nn.Identity()
        }

        if self.activation not in self.activation_map:
            raise ValueError(f"Unsupported activation function: {self.activation}")
            

        # Multilayer Perceptron 
        self.mlp = nn.Sequential(
            nn.Linear(d_hidden, d_mlp),
            self.activation_map[self.activation](), 
            nn.Dropout(dropout),
            nn.Linear(d_mlp, d_hidden)
        )
        
    def forward(self, x): 
        # Pre-Norm Multi-Head Attention 
        norm_x = self.norm1(x) 
        attn_output = self.attention(norm_x)  
        x = x + self.dropout1(attn_output)
        
        # Post-Norm Feed Forward Network
        norm_x = self.norm2(x)  
        mlp_output = self.mlp(norm_x)
        x = x + self.dropout2(mlp_output)  
        return x

"""Transformer Encoder with DropPath - Stochastic Depth"""
# Used for Swin Transformer Experiments
class TransformerEncoder_DropPath(nn.Module): 
    def __init__(self, args, d_hidden, d_mlp, num_heads, dropout, attention_dropout, drop_path):
        super(TransformerEncoder_DropPath, self).__init__()
        self.args = args 

        self.d_hidden = d_hidden 
        self.d_mlp = d_mlp
        self.num_heads = num_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout

        self.attention = MultiHeadAttention(d_hidden, num_heads, attention_dropout)

        self.norm1 = nn.LayerNorm(d_hidden)
        self.norm2 = nn.LayerNorm(d_hidden)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

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
            "hardtanh": lambda: nn.Hardtanh(inplace=args.inplace), 
            "identity": lambda: nn.Identity()
        }

        if self.activation not in self.activation_map:
            raise ValueError(f"Unsupported activation function: {self.activation}")
            

        # Multilayer Perceptron 
        self.mlp = nn.Sequential(
            nn.Linear(d_hidden, d_mlp),
            self.activation_map[self.activation](), 
            nn.Dropout(dropout),
            nn.Linear(d_mlp, d_hidden)
        )
        
    def forward(self, x): 
        # Pre-Norm Multi-Head Attention 
        norm_x = self.norm1(x) 
        attn_output = self.attention(norm_x)  
        x = x + self.drop_path(attn_output)
        
        # Post-Norm Feed Forward Network
        norm_x = self.norm2(x)  
        mlp_output = self.mlp(norm_x)
        x = x + self.drop_path(mlp_output)  
        return x

"""Also can use timm's DropPath Implementation"""
# from timm.models.layers import DropPath
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""
    def __init__(self, drop_prob: float = 0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        
    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output


"""BatchNorm for ViT"""
class ViTBatchNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        # We use BatchNorm1d because our input is a 1D sequence of tokens
        self.bn = nn.BatchNorm1d(num_features, eps=eps, momentum=momentum)

    def forward(self, x):
        # Incoming shape: (B, N, C)
        
        # 1. Permute to (B, C, N) for BatchNorm
        x = x.transpose(1, 2) 
        
        # 2. Apply Batch Normalization
        x = self.bn(x)
        
        # 3. Permute back to (B, N, C) for the Attention/MLP blocks
        x = x.transpose(1, 2) 
        
        return x