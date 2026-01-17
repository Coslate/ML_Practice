
# Question
# Write a Transformer Encoder Layer with GQA without for-loop / without einsum
# X: [N, T, D]
# use pre-Norm
# reference check uses PyTorch built-in GQA in scaled_dot_product_attention (enable_gqa=True)

import torch
import math
import torch.nn.functional as F

def softmax_lastdim(z: torch.Tensor) -> torch.Tensor:
    z = z - z.max(dim=-1, keepdim=True).values
    z = torch.exp(z)
    return z / z.sum(dim=-1, keepdim=True)

def layer_norm_torch(Z: torch.tensor, eps: float) -> torch.Tensor:
    mean = Z.mean(dim=-1, keepdim=True)
    var = Z.var(dim=-1, keepdim=True)
    return (Z - mean) / torch.sqrt(var + eps)

def relu_torch(Z: torch.Tensor) -> torch.Tensor:
    return torch.relu(Z)

def gqa_attention_torch(
        X: torch.Tensor, #[N, T, D]
        mask: torch.Tensor, #[T, T]
        n_q: int, #number of query heads
        n_kv: int, #number of kv heads
        W_Q: torch.Tensor, #[D, D], projects to n_q*dk = D
        W_K: torch.Tensor, #[D, n_kv*dk]
        W_V: torch.Tensor, #[D, n_kv*dk]
        W_out: torch.Tensor, #[D, D]
) -> torch.Tensor:
    assert X.ndim == 3
    N, T, D = X.shape
    assert D % n_q == 0
    assert n_q % n_kv == 0
    dk = D // n_q
    group = n_q // n_kv

    assert W_Q.shape == (D, D)
    assert W_K.shape == (D, n_kv*dk)
    assert W_V.shape == (D, n_kv*dk)
    assert W_out.shape == (D, D)

    # projections
    q = X @ W_Q #[N, T, D]
    k = X @ W_K # [N, T, n_kv*dk]
    v = X @ W_V #[N, T, n_kv*dk]

    # reshape
    q = q.reshape(N, T, n_q, dk).transpose(1, 2).contiguous() #[N, n_q, T, dk]
    k = k.reshape(N, T, n_kv, dk).transpose(1, 2).contiguous() #[N, n_kv, T, dk]
    v = v.reshape(N, T, n_kv, dk).transpose(1, 2).contiguous() #[N, n_kv, T, dk]

    # group queries
    qg = q.reshape(N, n_kv, group, T, dk)

    # attn scores
    att = (qg @ k.transpose(-1, -2).unsqueeze(2)) / math.sqrt(dk) # [N, n_kv, group, T, dk] @ [N, n_kv, 1, dk, T]
    if mask is not None:
        att = att + mask #[N, n_kv, group, T, T]

    att = softmax_lastdim(att)

    # weighted sum
    y = att @ v.unsqueeze(2) # [N, n_kv, group, T, T] @  [N, n_kv, 1, T, dk] = [N, n_kv, group, T, dk]

    # merge heads back: [N, n_q, T, dk] -> [N, T, D]
    y = y.reshape(N, n_q, T, dk)
    y = y.transpose(1, 2).reshape(N, T, D)

    # output projection
    out = y @ W_out #[N, T, D]

    return out


def transformer_block_gqa_torch(
        X: torch.Tensor, #[N, T, D]
        mask: torch.Tensor, #[T, T]
        n_q: int,
        n_kv: int,
        W_Q: torch.Tensor, #[D, D]
        W_K: torch.Tensor, #[D, n_kv*dk]
        W_V: torch.Tensor, #[D, n_kv*dk]
        W_out: torch.Tensor, #[D, D]
        W_ff1: torch.Tensor, #[D, d_ff]
        W_ff2: torch.Tensor, #[d_ff, D]
        eps: float
) -> torch.Tensor:
    #pre-norm
    Z = X + gqa_attention_torch(layer_norm_torch(X, eps), mask, n_q, n_kv, W_Q, W_K, W_V, W_out)
    ffn = relu_torch(layer_norm_torch(Z, eps) @ W_ff1) @ W_ff2
    out = Z + ffn
    return out

# reference check vs pytorch
torch.manual_seed(0)
N, T, D = 10, 100, 64
n_q, n_kv = 8, 2
assert D % n_q == 0 and n_q % n_kv == 0
dk = D//n_q
group = n_q // n_kv
d_ff = 64*4

X = torch.randn(N, T, D)
M = torch.triu(-float('inf')*torch.ones(T, T), diagonal=1)

# weights (no bias)
W_Q = torch.randn(D, D) * 0.02
W_K = torch.randn(D, n_kv*dk) * 0.02
W_V = torch.randn(D, n_kv*dk) * 0.02
W_out = torch.rand(D, D) * 0.02
W_ff1 = torch.randn(D, d_ff) * 0.02
W_ff2 = torch.randn(d_ff, D) * 0.02
eps = 1e-5

# Out output
Y = transformer_block_gqa_torch(X, M, n_q, n_kv, W_Q, W_K, W_V, W_out, W_ff1, W_ff2, eps)

# Reference output
Xn = layer_norm_torch(X, eps) #[N, T, D]

q = (Xn @ W_Q).reshape(N, T, n_q, dk).transpose(1, 2).contiguous() #[N, n_q, T, dk]
k = (Xn @ W_K).reshape(N, T, n_kv, dk).transpose(1, 2).contiguous() #[N, n_kv, T, dk]
v = (Xn @ W_V).reshape(N, T, n_kv, dk).transpose(1, 2).contiguous() #[N, n_kv, T, dk]

try:
    y_heads = F.scaled_dot_product_attention(
        q, k, v,
        attn_mask = M, drouput_p = 0.0, is_causal=False,
        enable_gqa=True
    ) #[N, n_q, T, dk]
except TypeError:
    print(f"Use repeat_iterleave()!")
    k_rep = k.repeat_interleave(group, dim=1) #[N, n_q, T, dk]
    v_rep = v.repeat_interleave(group, dim=1) #[N, n_q, T, dk]
    y_heads = F.scaled_dot_product_attention(
        q, k_rep, v_rep,
        attn_mask=M, dropout_p=0.0, is_causal=False
    )

y_ref_attn = y_heads.transpose(1, 2).reshape(N, T, D) @ W_out #[N, T, D]
Z_ref = X + y_ref_attn
ffn_ref = relu_torch(layer_norm_torch(Z_ref, eps) @ W_ff1) @ W_ff2
Y_ref = Z_ref + ffn_ref

print(torch.norm(Y - Y_ref).item())
assert torch.allclose(Y, Y_ref, rtol=0, atol=1e-5), "should be close to Pytorch SDPA GQA reference"
print(f"OK")