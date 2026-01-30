
#Question: Given scores: [B, N], return topk values and indices per batch: vals: [B, K], idx: [B, K].
import torch
def topk_per_row(scores: torch.Tensor, k: int):
    # scores: [B, N]
    # vals: [B, K]
    # idx: [B, K]
    vals, idx = torch.topk(scores, k=k, dim=1, largest=True, sorted=True)
    return vals, idx

#quick check
torch.manual_seed(0)
scores = torch.randn(3, 10)
vals, idx = topk_per_row(scores, k=4)
assert vals.shape == (3, 4)
assert idx.shape == (3, 4)


#reference check vs torch.topk directly
vals_ref, idx_ref = torch.topk(scores, k=4, dim=1, largest=True, sorted=True)
assert torch.allclose(vals, vals_ref, atol=1e-5)
assert torch.allclose(idx, idx_ref, atol=1e-5)

torch.testing.assert_close(
    vals, vals_ref,
    rtol=0, atol=1e-5,
    msg=f"should be close"
)

print(f"OK")