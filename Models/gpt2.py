"""
GPT2 Model implementation in PyTorch. 

Original Code from OpenAI's GPT-2 repository, written in TensorFlow. 

https://github.com/openai/gpt-2/blob/master/src/model.py
"""

import math
import torch 
import torch.nn as nn 
import torch.nn.functional as F 

# Activation Functions 
from Models.activation import (GELU_s, SiLU_s, ZiLU_Old, ArcTan,
                               ArcTan_Approx, ZiLU, ZiLU_Approx)

class GPT2(nn.Module):
    def __init__(self, 
                 args, 
                 vocab_size=50257, 
                 max_seq_length=1024,
                 embedding_dim=768,
                 num_attention_heads=12,
                 num_layers=12,
                 device='cuda'):

        super(GPT2, self).__init__()

        self.args = args 

        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.embedding_dim = embedding_dim
        self.num_attention_heads = num_attention_heads
        self.num_layers = num_layers
        self.device = device

        # Embeddings 
        self.token_embeddings = nn.Embedding(vocab_size, embedding_dim)     
        self.position_embeddings = nn.Embedding(max_seq_length, embedding_dim)
        
        # Transformer Blocks    
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(args, embedding_dim, num_attention_heads, embedding_dim * 4, max_seq_length)
            for _ in range(num_layers)
        ])

        # Final Layer Norm 
        self.layer_norm = nn.LayerNorm(embedding_dim)

        # Linear output layer 
        self.lm_head = nn.Linear(embedding_dim, vocab_size, bias=False)
        
        # Initialize Weights 
        self.apply(self._init_weights)

        self.to(device)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    

    def forward(self, input_ids, target=None):
        batch_size, seq_length = input_ids.size()

        # Create position ids 
        position_ids = torch.arange(0, seq_length, dtype=torch.long, device=self.device)

        # Embeddings 
        token_embeds = self.token_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        x = token_embeds + position_embeds

        # Transformer Blocks 
        for layer in self.transformer_blocks:
            x = layer(x)

        # Final Layer Norm 
        x = self.layer_norm(x)

        if target is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None 

        return logits, loss
    
    def parameter_count(self, non_embeddings=True): 
        total_params = sum(p.numel() for p in self.parameters())
        if non_embeddings:
            total_params -= self.token_embeddings.weight.numel() + self.position_embeddings.weight.numel()
        return total_params
        
class CausalMultiHeadAttention(nn.Module):
    def __init__(self, d_embeddings, num_heads, max_seq_length):
        super(CausalMultiHeadAttention, self).__init__()

        assert d_embeddings % num_heads == 0, "Match Embeddings with Number of Heads"

        self.d_embedding = d_embeddings
        self.num_heads = num_heads
        self.max_seq_length = max_seq_length
        self.d_heads = d_embeddings // num_heads


        self.w_k = nn.Linear(d_embeddings, d_embeddings)
        self.w_q = nn.Linear(d_embeddings, d_embeddings)
        self.w_v = nn.Linear(d_embeddings, d_embeddings)
        self.w_o = nn.Linear(d_embeddings, d_embeddings)

        causal_mask = torch.tril(torch.ones(max_seq_length, max_seq_length)).view(1, 1, max_seq_length, max_seq_length)
        self.register_buffer('causal_mask', causal_mask)

    def split_heads(self, x):
        batch_size, seq_length, d_embeddings = x.shape 
        return x.view(batch_size, seq_length, self.num_heads, self.d_heads).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, num_heads, seq_length, d_heads = x.shape 
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, num_heads * d_heads)        
        
    def forward(self, x):
        k = self.split_heads(self.w_k(x))
        q = self.split_heads(self.w_q(x))
        v = self.split_heads(self.w_v(x))        

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / math.sqrt(v.size(-1)))

        seq_length = attn_scores.size(-2)
        mask_slice = self.causal_mask[:, :, :seq_length, :seq_length]
        attn_scores = attn_scores.masked_fill(mask_slice == 0, float('-inf'))

        # Softmax and weighting 
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_probs, v)

        attn_output = self.combine_heads(attn_output)
        attn_output = self.w_o(attn_output)
        return attn_output

class MLP(nn.Module):
    def __init__(self, args, d_model, d_ff):
        super(MLP, self).__init__() 

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
            "zilu_approx": lambda: ZiLU_Approx(sigma=args.sigma) 
        }

        if self.activation not in self.activation_map:
            raise ValueError(f"Unsupported activation function: {self.activation}")
            
        
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.activation_function = self.activation_map[self.activation]()

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, args, d_model, num_heads, d_ff, max_seq_length):
        super(TransformerBlock, self).__init__()
        self.attention = CausalMultiHeadAttention(d_model, num_heads, max_seq_length)
        self.mlp = MLP(args, d_model, d_ff)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Pre-Norm Multi-Head Attention 
        norm_x = self.layer_norm1(x)
        attn_output = self.attention(norm_x)
        x = x + attn_output

        # Post-Norm MLP
        norm_x = self.layer_norm2(x)
        mlp_output = self.mlp(norm_x)
        x = x + mlp_output
        return x


if __name__ == "__main__":
    model = GPT2(device='cpu')
    total_params = model.parameter_count()
    print(f"Total Parameters: {total_params}")

    ex = torch.randint(0, 50257, (2, 10)).long()
    logits, loss = model(ex)
    print("Logits shape:", logits.shape)