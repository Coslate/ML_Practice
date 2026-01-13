
# Question: Given x: [B, N, D] and idx: [B, K], gather to out: [B, K, D] without loops.
import torch

def batched_gather(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    #x: [B, N, D]
    #idx: [B, K]
    B, N, D = x.shape
    K = idx.shape[1]
    idx_expanded = idx[:, :, None].expand(B, K, D) #[B, K, D] , broadcasting to D of last dimenstion
    #idx_expanded = idx[:, :, None].repeat(1, 1, D) #[B, K, D], real copying D times of last dimension

    out = x.gather(dim=1, index=idx_expanded) #[B, K, D] out[b, k, d]=out[b, idx_expanded[b, k, d], d]
    return out


# quick check
torch.manual_seed(0)
x = torch.randn(2, 7, 4)
idx = torch.tensor([
    [0, 3, 6],
    [2, 2, 5]
], dtype=torch.long)
y = batched_gather(x, idx)
assert y.shape == (2, 3, 4)

# reference check
y_ref = torch.stack([x[b, idx[b]] for b in range(2)], dim=0)
#print(f"x[0, 1, 0:4] = {x[0, 1, 0:4]}")
#print(f"x[0, 1, 0:2] = {x[0, 1, 0:2]}")
#print(f"x[0, 1] = {x[0, 1]}")

assert torch.allclose(y, y_ref, atol=1e-6)
torch.testing.assert_close(
    y, y_ref,
    rtol=0, atol=1e-6,
    msg=f"should be closed to y_ref."
)
print(f"OK")