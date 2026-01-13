
# Question: L2 normalize x along the last axis with epsilon clamp to avoid NaNs (including all-zero rows).

import torch
import torch.nn.functional as F

def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    # x = [..., D]
    denom = torch.sqrt((x*x).sum(dim=-1, keepdim=True)).clamp_min(eps)
    return x/denom


# quick check
torch.manual_seed(0)
x = torch.randn(6, 5)
y = l2_normalize(x)
assert y.shape == (6, 5), f"should be same shape"
assert torch.isfinite(y).all(), f"shold be finite"
assert torch.allclose(torch.sqrt((y*y).sum(dim=1)), torch.ones(6), atol=1e-5)

torch.testing.assert_close(
    torch.sqrt((y*y).sum(dim=1)), torch.ones(6),
    rtol=0, atol=1e-5,
    msg=f"Sim of last normalized dimension should be 1."
)

# zero-row safety
z = torch.zeros(3, 5)
yz = l2_normalize(z)
y_ref  = F.normalize(z, p=2, dim=-1, eps=1e-12)      # builtin

assert yz.shape == y_ref.shape, f"shape should be same as y_ref."
assert torch.isfinite(yz).all()
assert torch.allclose(yz, torch.zeros_like(yz))
assert torch.allclose(yz, y_ref, atol=1e-6, rtol=0)

torch.testing.assert_close(
    yz, torch.zeros_like(yz),
    rtol=0.0, atol=1e-6,
    msg="Should be all zeros."
)

torch.testing.assert_close(
    yz, y_ref,
    rtol=0.0, atol=1e-6,
    msg = f"Should be same as y_ref."
)
print(f"OK")