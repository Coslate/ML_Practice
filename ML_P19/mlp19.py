
# Question
# Write a Transformer Encoder Layer without for-loop
# X: [N, T, D]
# heads: scalar
# use pre-Norm

import numpy as np
import torch
import math
import torch.nn as nn

def softmax_lastdim(z: torch.Tensor) -> torch.Tensor:
    z = z-z.max(dim=-1, keepdim=True).values
    z = torch.exp(z)
    return z/z.sum(dim=-1, keepdim=True)


def multihead_attention_torch(
        X: torch.Tensor, #[N, T, d]
        mask: torch.Tensor, # [T, T], broadcasting to [N, heads, T, T], additive: 0 keep, -inf blocks
        heads: int,
        W_QKV: torch.Tensor, #[T, 3d]
        W_out: torch.Tensor  #[d, d_out] (often [d, d])
):
    assert X.ndim == 3, "X must be [N, T, d]"
    N, T, d = X.shape
    assert d % heads == 0, "d must be divisible by heads"
    dh = d // heads

    assert W_QKV.shape == (d, 3*d), f"W_QKV must be {(d, 3*d)}, got {tuple(W_QKV.shape)}"
    assert W_out.shape[0] == d, f"W_out first dim must be d = {d}, got {W_out.shape[0]}"

    # X@W_QKV -> [N, T, 3d]
    XW = X @ W_QKV

    # split last dim into K, Q, V, each [N, T, d]
    Q, K, V = XW.split(d, dim=-1)

    # reshape to heads: [N, T, heads, dh] then [N, heads, T, dh]
    Q = Q.reshape(N, T, heads, dh).transpose(1, 2) #[N, heads, T, dh]
    K = K.reshape(N, T, heads, dh).transpose(1, 2) #[N, heads, T, dh]
    V = V.reshape(N, T, heads, dh).transpose(1, 2) #[N, heads, T, dh]

    # scores: [N, heads, T, T]
    scores = Q @ K.transpose(-1, -2) #[N, heads, T, T]
    scores /= math.sqrt(dh)
    if mask is not None:
        scores = scores + mask


    # attn: [N, heads, T, T]
    attn = softmax_lastdim(scores)

    # context: [N, heads, T, dh]
    ctx = attn @ V

    # reshape -> [N, T, heads, dh] -> [N, T, d]
    ctx = ctx.transpose(1, 2).reshape(N, T, d)

    # out projection: [N, T, d_out]
    out = ctx @ W_out
    return out, attn

def layer_norm_torch(Z: torch.Tensor, eps: float) -> torch.Tensor:
    # matches Numpy: (Z-mean) / sqrt(var+eps)
    # use unbiased=False to amtch the np.var
    mean = Z.mean(dim=-1, keepdim=True)
    var = Z.var(dim=-1, keepdim=True, unbiased=False)
    return (Z-mean)/torch.sqrt(var + eps)

def relu_torch(Z: torch.Tensor) -> torch.Tensor:
    return torch.relu(Z)

def transformer_block_torch(
        X: torch.Tensor, #[N, T, d]
        mask: torch.Tensor, #[T, T], broadcasting to [N, heads, T, T]
        heads: int,
        W_QKV: torch.Tensor, #[d, 3d] 
        W_out: torch.Tensor, #[d, d] or [d, d_out]
        W_ff1: torch.Tensor, #[d, d_ff]
        W_ff2: torch.Tensor, #[d_ff, d]
        eps: float
) -> torch.Tensor:
    # pre-Norm
    Z = X + multihead_attention_torch(layer_norm_torch(X, eps), mask, heads, W_QKV, W_out)[0] #[N, T, d]
    ffn = relu_torch(layer_norm_torch(Z, eps) @ W_ff1) @ W_ff2 #[N, T, d]
    out = Z + ffn #[N, T, d]
    return out



#reference check vs pytorch version
N, T, d = 10, 100, 64
heads = 4
X = torch.randn(N, T, d)
M = torch.triu(-float('inf')*torch.ones((T, T)), 1)

trans = nn.TransformerEncoderLayer(d, heads, dim_feedforward=128, dropout=0.0, batch_first=True, norm_first=True)
trans.self_attn.in_proj_bias.data.zero_()
trans.self_attn.out_proj.bias.data.zero_()
trans.linear1.bias.data.zero_()
trans.linear2.bias.data.zero_()
trans.norm1.weight.data.fill_(1.0)
trans.norm1.bias.data.zero_()
trans.norm2.weight.data.fill_(1.0)
trans.norm2.bias.data.zero_()
Y_ref = trans(X, M)

Y = transformer_block_torch(X, M, heads,
                            trans.self_attn.in_proj_weight.T,
                            trans.self_attn.out_proj.weight.T,
                            trans.linear1.weight.T,
                            trans.linear2.weight.T,
                            trans.norm1.eps)

print(np.linalg.norm(Y.detach() - Y_ref.detach()))
assert torch.allclose(Y, Y_ref, rtol=0, atol=1e-5), f"should be close to pyrotch version."
print(f"OK")
