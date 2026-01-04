
# Question: Given x: [B, N, D] and mask: [B, N] (bool or 0/1), compute masked mean out: [B, D] with safe divide (no NaNs when a row has zero valid entries).
import torch

def masked_mean(x: torch.Tensor, mask: torch.Tensor, eps: float=1e-12) -> torch.Tensor:
    # x: [B, N, D]
    # mask: [B, N]
    m = mask.to(dtype=x.dtype) #[B, N]
    x_sum = (x*m[:, :, None]).sum(dim=1) #[B, D]
    denom = m.sum(dim=1).clamp_min(eps) #[B]
    return x_sum/denom[:, None] #[B, D]


# quick check
torch.manual_seed(0)
x = torch.randn(4, 6, 3)
mask = torch.tensor([
    [1, 1, 0, 0, 0, 0],
    [1, 0, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 0], # all-masked row
    [1, 1, 1, 1, 1, 1],
], dtype=torch.bool)

y = masked_mean(x, mask)
assert y.shape == (4, 3)
assert torch.isfinite(y).all()
print(f"torch.isfinite(y) = {torch.isfinite(y)}")

# reference check vs anive
y_ref = torch.zeros(4, 3)

for b in range(4):
    idx = mask[b].nonzero(as_tuple=False).squeeze(-1) #[M, 1] -> [M], M is the number of nonzero elements in mask
    #print(f"mask[{b}] = {mask[b]}")
    #print(f"mask[{b}].nonzero() = {mask[b].nonzero()}")
    #print(f"mask[{b}].nonzero(as_tuple=False) = {mask[b].nonzero(as_tuple=False)}")
    #print(f"mask[{b}].nonzero(as_tuple=True) = {mask[b].nonzero(as_tuple=True)}")
    #print(f"mask[{b}].nonzero(as_tuple=False).shape = {mask[b].nonzero(as_tuple=False).shape}")
    #print(f"idx.numel() = {idx.numel()}")
    y_ref[b] = x[b, idx].mean(dim=0) if idx.numel() > 0 else 0.0 # x[b, idx] = [M, D] -> [D]
    #print(f"x[{b}, {idx}].shape = {x[b, idx].shape}")

assert torch.allclose(y, y_ref, atol=1e-6)

print(f"OK")
