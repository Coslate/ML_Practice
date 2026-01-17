
# Question
# Implement RoPE (Rotary Positional Embeddings) without for-loop / without einsum()
# Y: [B, H, T, D] where D is even
# Reference check: compare against an explicit rotation-matrix implementation (PyTorch ops only)
# and assert torch.allclose()

import torch
import torch.nn as nn

class RotaryPositionalEmbeddings(nn.Module):
    def __init__(self, d: int, base: int=10_000):
        super().__init__()
        assert d % 2 == 0
        self.d = d
        self.base = base

        rope_theta = 1.0/base**(torch.arange(0, d, 2, dtype=torch.float32)/d) #[d/2]
        self.register_buffer("rope_theta", rope_theta, persistent=False)

        self.cache_t = 0
        self.register_buffer("cos_cached", torch.empty(0), persistent=False)
        self.register_buffer("sin_cached", torch.empty(0), persistent=False)

    def _build_cache(self, Y: torch.Tensor):
        # Y: [B, H, T, D]
        B, H, T, D = Y.shape
        assert D == self.d

        if self.cache_t >= T and self.cos_cached.numel() != 0:
            return

        # positions: 1..T
        pos = torch.arange(1, T+1, device=Y.device, dtype=self.rope_theta.dtype)
        freqs = torch.outer(pos, self.rope_theta) #[T, D/2]

        # expand to [T, D] as [theta0, theta0, theta1, theta1, ...]
        cap_c = torch.repeat_interleave(freqs, 2, dim=-1) #[T, D]

        cos = cap_c.cos()[None, None, :, :].to(dtype=Y.dtype) #[1, 1, T, D]
        sin = cap_c.sin()[None, None, :, :].to(dtype=Y.dtype) #[1, 1, T, D]

        self.cos_cached = cos
        self.sin_cached = sin
        self.cache_t = T

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        #x: [..., D] -> [..., -x1, x0, -x3, x2, ...]
        x_even = x[..., ::2] #[..., D/2]
        x_odd = x[..., 1::2] #[..., D/2]
        return torch.stack((-x_odd, x_even), dim=-1).reshape_as(x)

    def forward(self, Y:torch.Tensor) -> torch.Tensor:
        #Y: [B, H, T, D]
        assert Y.ndim == 4
        B, H, T, D = Y.shape
        assert D == self.d

        self._build_cache(Y)
        cos = self.cos_cached[:, :, :T, :] #[1, 1, T, D]
        sin = self.sin_cached[:, :, :T, :] #[1, 1, T, D]
        return Y*cos + self._rotate_half(Y)*sin


def rope_reference_matmul(Y: torch.Tensor, d: int, base: int = 10_000) -> torch.Tensor:
    '''
    Explicit rotation-matrix refereence
    Y: [B, H, T, D]
    '''

    assert Y.ndim == 4
    B, H, T, D = Y.shape
    assert D == d and D % 2== 0
    d2 = D//2

    rope_theta = 1.0 / (base ** (torch.arange(0, D, 2, device=Y.device, dtype=torch.float32) / D)) #[d2]
    pos = torch.arange(1, T+1, device=Y.device, dtype=torch.float32) #[T]
    freqs = torch.outer(pos, rope_theta) #[T, d2]
    c = freqs.cos().to(Y.dtype) #[T, d2]
    s = freqs.sin().to(Y.dtype) #[T, d2]

    # BUild 
    R = torch.stack([
        torch.stack((c, -s), dim=-1), #[T, d2, 2]
        torch.stack((s, c), dim=-1) #[T, d2, 2]
    ],
    dim=-2) #[N, d2, 2, 2]

    # Pair Y into (..., d2, 2)
    Yp = Y.view(B, H, T, d2, 2)
    vec = Yp.unsqueeze(-1) #[B, H, T, d2, 2, 1]
    R = R.unsqueeze(0).unsqueeze(0) #[1, 1, T, d2, 2, 2]
    out = (R @ vec).squeeze(-1) #[B, H, T, d2, 2]
    return out.reshape(B, H, T, D)

# reference check
torch.manual_seed(0)
B, H, T, D  = 2, 4, 16, 64
Y = torch.randn(B, H, T, D, dtype=torch.float32)
rope = RotaryPositionalEmbeddings(d=D, base=10_000)
Y_rope = rope(Y)
Y_ref = rope_reference_matmul(Y, d=D, base=10_000)
print(torch.norm(Y_rope-Y_ref).item())
assert torch.allclose(Y_rope, Y_ref, rtol=0, atol=1e-5), "RoPE should match reference."
print("OK")
