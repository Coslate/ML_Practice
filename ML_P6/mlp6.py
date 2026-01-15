
# Question: Implement stable softmax for logits: [B, C]; verify each row sums to 1 and remains finite for large logits.
import torch

def stable_softmax(logits: torch.Tensor, dim: int=-1) -> torch.Tensor:
    # Logits: [..., C]
    z = logits - logits.max(dim=dim, keepdim=True).values #[..., C]
    #z = logits
    expz = torch.exp(z)
    return expz/expz.sum(dim=dim, keepdim=True)

# quick check
torch.manual_seed(0)
logits = torch.randn(4, 7)*80000000
p = stable_softmax(logits, dim=1)
assert p.shape == (4, 7)
assert torch.isfinite(p).all()
assert torch.allclose(p.sum(dim=1), torch.ones((4,)), atol=1e-5)

# reference check
p_ref = torch.softmax(logits, dim=1)
assert torch.allclose(p, p_ref, atol=1e-5, rtol=1e-5)
print(f"OK")