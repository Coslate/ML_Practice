
# Question: Implement stable softmax for logits: [B, C]; verify each row sums to 1 and remains finite for large logits.

import torch

def stable_softmax(logits: torch.Tensor, dim: int=-1) -> torch.Tensor:
    # logits: [..., C]
    z = logits - logits.max(dim=dim, keepdim=True).values #[..., C]
    expz = torch.exp(z)
    return expz/expz.sum(dim=dim, keepdim=True)

# quick check
torch.manual_seed(0)
logits = torch.randn(4, 7) * 80.0
p = stable_softmax(logits, dim=1)
assert p.shape == (4, 7)
assert torch.isfinite(p).all()
assert torch.allclose(p.sum(dim=1), torch.ones((4,)), atol=1e-6)

# reference check vs torch.softmax
p_ref = torch.softmax(logits, dim=1)
assert torch.allclose(p, p_ref, atol=1e-6)
torch.testing.assert_close(
    p, p_ref,
    rtol=0, atol=1e-6,
    msg=f"p should be close to p_ref."
)
print(f"OK")