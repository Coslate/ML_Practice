
#Question: Given scores: [B, N], return topk values and indices per batch: vals: [B, K], idx: [B, K].

import torch

def topk_per_row(scores: torch.Tensor, k: int) -> torch.Tensor:
    # scores: [B, N]
    # vals: [B, K]
    # idx: [B, K]

    vals, idx = torch.topk(scores, k=k, dim=1, largest=True, sorted=True)
    return vals, idx

# quick check
torch.manual_seed(0)
scores = torch.randn(3, 10)
vals, idx = topk_per_row(scores, k=4)
assert vals.shape == (3, 4)
assert idx.shape == (3, 4)

torch.testing.assert_close(
    vals.shape, (3, 4),
    rtol=0, atol=0,
    msg = f"vals.shape should be (3, 4), but got {vals.shape} instead."
)

torch.testing.assert_close(
    idx.shape, (3, 4),
    rtol=0, atol=0,
    msg = f"idx.shape should be (3, 4), but got {idx.shape} instead."
)


vals_ref, idx_ref = torch.topk(scores, 4, dim=1, largest=True, sorted=True)
assert torch.allclose(vals, vals_ref, atol=1e-6)
assert torch.equal(idx, idx_ref)

torch.testing.assert_close(
    vals, vals_ref,
    rtol=0, atol=1e-6,
    msg = f"vals differ from vals_ref."
)

torch.testing.assert_close(
    idx, idx_ref,
    rtol=0, atol=0,
    msg = f"idx should be equal to idx_ref."
)

print(f"OK")



