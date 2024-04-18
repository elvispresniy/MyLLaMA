import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from dataclasses import dataclass

@dataclass
class ModelArgs:
    max_batch_size: int
    max_seq_len: int
    dim: int
    vocab_size: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    ffn_hidden_dim: int
    device: str
    norm_eps: float = 1e-5
    att_dropout_p: float = 0.1
    ffn_dropout_p: float = 0.1
    embed_dropout_p: float = 0.1

# Credits for RoPE: https://github.com/hkproj/pytorch-llama/blob/main/model.py#L257
class RoPE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.args = args
        self.freqs_complex = self.precompute_theta_pos_frequencies(args.head_dim, args.max_seq_len, device=self.args.device)

    def precompute_theta_pos_frequencies(self, head_dim: int, seq_len: int, device: str, theta: float = 10000.0):
        # As written in the paragraph 3.2.2 of the paper
        # >> In order to generalize our results in 2D to any xi ∈ Rd where **d is even**, [...]
        assert head_dim % 2 == 0, "Dimension must be divisible by 2"
        # Build the theta parameter
        # According to the formula theta_i = 10000^(-2(i-1)/dim) for i = [1, 2, ... dim/2]
        # Shape: (Head_Dim / 2)
        theta_numerator = torch.arange(0, head_dim, 2).float()
        # Shape: (Head_Dim / 2)
        theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device) # (Dim / 2)
        # Construct the positions (the "m" parameter)
        # Shape: (Seq_Len)
        m = torch.arange(seq_len, device=device)
        # Multiply each theta by each position using the outer product.
        # Shape: (Seq_Len) outer_product* (Head_Dim / 2) -> (Seq_Len, Head_Dim / 2)
        freqs = torch.outer(m, theta).float()
        # We can compute complex numbers in the polar form c = R * exp(m * theta), where R = 1 as follows:
        # (Seq_Len, Head_Dim / 2) -> (Seq_Len, Head_Dim / 2)
        freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_complex

    def apply_rotary_embeddings(self, x: torch.Tensor, freqs_complex: torch.Tensor, device: str):
        # Separate the last dimension pairs of two values, representing the real and imaginary parts of the complex number
        # Two consecutive values will become a single complex number
        # (B, Seq_Len, H, Head_Dim) -> (B, Seq_Len, H, Head_Dim/2)
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        # Reshape the freqs_complex tensor to match the shape of the x_complex tensor. So we need to add the batch dimension and the head dimension
        # (Seq_Len, Head_Dim/2) --> (1, Seq_Len, 1, Head_Dim/2)
        freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
        # Multiply each complex number in the x_complex tensor by the corresponding complex number in the freqs_complex tensor
        # Which results in the rotation of the complex number as shown in the Figure 1 of the paper
        # (B, Seq_Len, H, Head_Dim/2) * (1, Seq_Len, 1, Head_Dim/2) = (B, Seq_Len, H, Head_Dim/2)
        x_rotated = x_complex * freqs_complex
        # Convert the complex number back to the real number
        # (B, Seq_Len, H, Head_Dim/2) -> (B, Seq_Len, H, Head_Dim/2, 2)
        x_out = torch.view_as_real(x_rotated)
        # (B, Seq_Len, H, Head_Dim/2, 2) -> (B, Seq_Len, H, Head_Dim)
        x_out = x_out.reshape(*x.shape)
        return x_out.type_as(x).to(device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        freqs_complex = self.freqs_complex[:seq_len]
        return self.apply_rotary_embeddings(x, freqs_complex, x.device)
    
class RMSNorm(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.eps = args.norm_eps
        self.g_params = nn.Parameter(torch.ones(args.dim))

    def forward(self, x: torch.Tensor):
        rms = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return rms * self.g_params
    
class FeedForward(nn.Module):
    '''FeedForward layer with SwiGLU activation.'''
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.w1 = nn.Linear(args.dim, args.ffn_hidden_dim, bias=False)
        self.w2 = nn.Linear(args.ffn_hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, args.ffn_hidden_dim, bias=False)

        self.dropout = nn.Dropout(args.ffn_dropout_p)

    def forward(self, x: torch.Tensor):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))
    
class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.args = args

        self.n_rep = args.n_heads // args.n_kv_heads

        self.pos_embed = RoPE(args)

        self.wq = nn.Linear(args.dim, args.n_heads * args.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)

        self.wo = nn.Linear(args.n_heads * args.head_dim, args.dim, bias=False)

        self.dropout = nn.Dropout(args.att_dropout_p)

        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, args.n_kv_heads, args.head_dim))
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, args.n_kv_heads, args.head_dim))

        self.register_buffer('causal_mask',
                        torch.triu(torch.ones([args.max_seq_len, args.max_seq_len],
                                            dtype=torch.bool), diagonal=1)
                        .view(1, 1, args.max_seq_len, args.max_seq_len))
        
    def expand_kv(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, n_heads, head_dim = x.shape
        if self.n_rep == 1:
            return x
        else:
            x = x[:, :, :, None, :].expand(batch_size, seq_len, n_heads, self.n_rep, head_dim)
            return x.reshape(batch_size, seq_len, n_heads * self.n_rep, head_dim)
        
    def clear_cache(self) -> None:
        self.cache_k = torch.zeros((self.args.max_batch_size, self.args.max_seq_len, self.args.n_kv_heads, self.args.head_dim))
        self.cache_v = torch.zeros((self.args.max_batch_size, self.args.max_seq_len, self.args.n_kv_heads, self.args.head_dim))

    def forward(self, x: torch.Tensor, mask: torch.BoolTensor|None=None) -> torch.Tensor:
        # x.shape = (batch_size, seq_len, dim)
        batch_size, seq_len, _ = x.shape

        # Apply linear layers
        # (batch_size, seq_len, embed_dim) -> (batch_size, seq_len, n_heads * head_dim)
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # Reshape resulting tensors
        # (batch_size, seq_len, n_heads * head_dim) -> (batch_size, seq_len, n_heads, head_dim)
        xq = xq.view(batch_size, seq_len, self.args.n_heads, self.args.head_dim)
        # (batch_size, seq_len, n_kv_heads * head_dim) -> (batch_size, seq_len, n_kv_heads, head_dim)
        xk = xk.view(batch_size, seq_len, self.args.n_kv_heads, self.args.head_dim)
        xv = xv.view(batch_size, seq_len, self.args.n_kv_heads, self.args.head_dim)

        # Apply Rotary Positional Embedding
        # (batch_size, seq_len, n_heads, head_dim) -> (batch_size, seq_len, n_heads, head_dim)
        xq = self.pos_embed(xq)
        # (batch_size, seq_len, n_kv_heads, head_dim) -> (batch_size, seq_len, n_kv_heads, head_dim)
        xk = self.pos_embed(xk)

        # Expand heads of keys and values to match n_head of queries
        # (batch_size, seq_len, n_kv_heads, head_dim) -> (batch_size, seq_len, n_heads, head_dim)
        xk = self.expand_kv(xk)
        xv = self.expand_kv(xv)

        # Transpose seq_len and n_heads to multiply xq by xk
        # (batch_size, seq_len, n_heads, head_dim) -> (batch_size, n_heads, seq_len, head_dim)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # Multiply queries and keys
        # (batch_size, n_heads, seq_len, head_dim) @ (batch_size, n_heads, head_dim, seq_len)
        # result: (batch_size, n_heads, seq_len, seq_len)
        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.args.head_dim)

        # Apply the mask
        combined_mask = self.causal_mask[:, :, :seq_len, :seq_len]
        if mask is not None:
            combined_mask += mask.view(batch_size, 1, 1, seq_len)
        scores.masked_fill_(combined_mask, float("-inf"))
        # Apply the softmax
        scores = scores.softmax(dim=-1)
        # Apply the dropout layer
        scores = self.dropout(scores)

        # Multiply by values matrix
        # (batch_size, n_heads, seq_len, seq_len) @ (batch_size, n_heads, seq_len, head_dim)
        # result: (batch_size, n_heads, seq_len, head_dim)
        output = torch.matmul(scores, xv)
        # (batch_size, n_heads, seq_len, head_dim) -> (batch_size, seq_len, n_heads * head_dim)
        output = output.transpose(1, 2).reshape(batch_size, seq_len, self.args.n_heads * self.args.head_dim)

        # (batch_size, seq_len, n_heads * head_dim) -> (batch_size, seq_len, dim)
        return self.wo(output)
    
    
class Encoder(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.norm1 = RMSNorm(args)
        self.norm2 = RMSNorm(args)

        self.attention = Attention(args)

        self.ffn = FeedForward(args)

    def forward(self, x: torch.Tensor, mask: torch.BoolTensor|None=None):
        h = x + self.attention(self.norm1(x), mask)
        out = h + self.ffn(self.norm2(x))
        return out
    
    def clear_cache(self):
        self.attention.clear_cache()

    
class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.embed_dropout = nn.Dropout(args.embed_dropout_p)
        
        self.norm = RMSNorm(args)

        self.layers = nn.ModuleList([
            Encoder(args) for _ in range(args.n_layers)
        ])

        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

    def rep_penalty(self, input_ids: torch.LongTensor, scores: torch.Tensor, penalty: float) -> torch.Tensor:
        mask = torch.zeros_like(scores, dtype=bool)
        mask.scatter_(-1, input_ids.unsqueeze(0), torch.ones_like(scores, dtype=bool))
        rep_scores = torch.masked_fill(scores, ~mask, 0)
        norm_scores = torch.masked_fill(scores, mask, 0)
        rep_scores = torch.where(rep_scores < 0, rep_scores * penalty, rep_scores / penalty)
        return norm_scores + rep_scores
    
    def _clear_cache(self):
        for layer in self.layers:
            layer.clear_cache()

    def forward(self, x: torch.Tensor, mask: torch.BoolTensor|None=None, penalty: float = 1.1):
        input_ids = x
        x = self.embed_dropout(self.tok_embeddings(x))

        for layer in self.layers:
            x = layer(x, mask)

        x = self.output(self.norm(x))
        x = self.rep_penalty(input_ids, x[:, -1:, :], penalty)

        return x