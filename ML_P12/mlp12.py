
# Question: Given x: [T, D], produce windowed tensor out: [T-W+1, W, D] efficiently.
import torch

def rolling_windows(x: torch.Tensor, W: int) -> torch.Tensor:
    #x: [T, D]
    #out: [T-W+1, W, D]
    #return x.unfold(dimension=0, size=W, step=1) #[T-W+1, D, W]
    return x.unfold(dimension=0, size=W, step=1).permute(0, 2, 1).contiguous() #[T-W+1, W, D]


# quick check
torch.manual_seed(0)
T, D, W = 8, 3, 4
x = torch.randn(T, D)
w = rolling_windows(x, W)
assert w.shape == (T-W+1, W, D)

# reference check vs naive
w_ref = torch.stack([x[i:i+W] for i in range(T-W+1)], dim=0)
assert torch.allclose(w, w_ref, atol=1e-6)
print(f"OK")