
# Question: Implement stable logsumexp for logits: [B, C] over dim=1, and match torch.logsumexp.
import torch

def stable_logsumexp(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    m = logits.max(dim=dim, keepdim=True).values #[B, 1]
    return (m + torch.log(torch.exp(logits-m).sum(dim=dim, keepdim=True))).squeeze(dim) #[B]


# quick check
torch.manual_seed(0)
logits = torch.randn(3, 9) * 80.0
lse = stable_logsumexp(logits, dim=1)
assert lse.shape == (3,)
assert torch.isfinite(lse).all()

# reference check vs torch.logsumexp
lse_ref = torch.logsumexp(logits, dim=1)
assert torch.allclose(lse, lse_ref, atol=1e-6)
torch.testing.assert_close(
    lse, lse_ref,
    rtol=0, atol=1e-6,
    msg=f"lse should be equal to lse_ref."
)
print(f"OK")