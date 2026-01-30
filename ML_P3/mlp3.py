
# Question: Given x: [B, N, D] and mask: [B, N] (bool or 0/1), compute masked mean out: [B, D] with safe divide (no NaNs when a row has zero valid entries).
import torch
def masked_mean(x: torch.Tensor, mask: torch.Tensor, eps: float=1e-12) -> torch.Tensor:
    #x: [B, N, D]
    #mask: [B, N]
    m = mask.to(dtype=x.dtype)
    x_sum = (x*m[:, :, None]).sum(dim=1) #[B, N, D] -> [B, N]
    denom = m.sum(dim=1).clamp_min(eps) #[B]
    return x_sum/denom[:, None] #[B, D]

#quick check
torch.manual_seed(0)
x = torch.randn(4, 6, 3)
mask = torch.tensor([
    [1, 1, 0, 0, 0, 0],
    [1, 0, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1]
], dtype=torch.bool)

y = masked_mean(x, mask)
assert y.shape == (4, 3)
assert torch.isfinite(y).all()

#reference check vs naive
y_ref = torch.zeros((4, 3))
for b in range(4):
    idx = mask[b].nonzero(as_tuple=False).squeeze(-1) #[M, 1] -> #[M]
    y_ref[b] = x[b, idx].mean(dim=0) if idx.numel() > 0 else 0.0 #[M, D] -> [D]
assert torch.allclose(y, y_ref, atol=1e-5)
print(f"OK")
