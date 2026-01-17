
#Question: Given values: [N] and bins: [N] in [0..K-1], compute sum_per_bin: [K] using scatter/add (no Python loops over N).
import torch

def scatter_sum(values: torch.Tensor, bins: torch.Tensor, K: int) -> torch.Tensor:
    # values: [N]
    # bins: [N] long in [0,...,K-1]
    out = values.new_zeros((K,))
    out.index_add_(dim=0, index=bins, source=values) #[K], out[index[i]] += values[i]
    return out

# quick check
torch.manual_seed(0)
values = torch.randn(20)
bins = torch.randint(0, 6, (20,), dtype=torch.long)
s = scatter_sum(values, bins, K=6)
assert s.shape == (6,)

# reference check vs bincount
s_ref = torch.bincount(bins, weights=values, minlength=6)
assert torch.allclose(s, s_ref, atol=1e-5)

print(f"OK")