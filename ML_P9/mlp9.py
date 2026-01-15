
# Question: L2 normalize x along the last axis with epsilon clamp to avoid NaNs (including all-zero rows).
import torch

def l2_normalize(x: torch.Tensor, eps: float=1e-12) -> torch.Tensor:
    # x: [..., D]
    denom = torch.sqrt((x*x).sum(dim=-1, keepdim=True)).clamp_min(eps)
    return x / denom #[..., D]


# quick check
torch.manual_seed(0)
x = torch.randn(6, 5)
y = l2_normalize(x)
assert y.shape == (6, 5)
assert torch.isfinite(y).all()
assert torch.allclose(torch.sqrt((y*y).sum(dim=1)), torch.ones(6), atol=1e-5)

# zero-row safety
z = torch.zeros((3, 5))
yz = l2_normalize(z)
assert torch.isfinite(yz).all()
assert torch.allclose(yz, torch.zeros_like(yz))

# reference check
y_ref = torch.nn.functional.normalize(x, p=2, dim=-1, eps=1e-12)
assert torch.allclose(y, y_ref, atol=1e-5)

print(f"OK")