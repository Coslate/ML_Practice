
# Question: Given features: [N, D] and group_id: [N] in [0..K-1], compute mean: [K, D] (no loops over N).
import torch

def groupby_mean(features: torch.Tensor, group_id: torch.Tensor, K:int, eps: float=1e-12) -> torch.Tensor:
    # feature: [N, D]
    # group_id: [N] Long in [0, ..., K-1]
    N, D = features.shape
    out_sum = features.new_zeros((K, D))
    out_sum.index_add_(dim=0, index=group_id, source=features) #[K, D], out_sum[grou_id[i]] += features[i]

    #ones = torch.ones((N,), device=features.device, dtype=features.dtype)
    ones = features.new_ones((N,)) #same device/dtype as features
    out_cnt = features.new_zeros((K,))
    out_cnt.index_add_(dim=0, index=group_id, source=ones) #[K], out_cnt[group_id[i]] += ones[i]

    return out_sum/out_cnt.clamp_min(eps)[:, None] #[K, D]


# quick check
torch.manual_seed(0)
features = torch.randn(10, 4)
group_id = torch.randint(0, 5, (10,), dtype=torch.long)
out = groupby_mean(features, group_id, 5)
assert out.shape == (5, 4)
assert torch.isfinite(out).all()

# reference check vs naive
out_ref = torch.zeros((5, 4))
cnt_ref = torch.zeros(5)

for i in range(10):
    g = int(group_id[i])
    out_ref[g] += features[i]
    cnt_ref[g] += 1
out_ref = out_ref/cnt_ref.clamp_min(1e-12)[:, None]

assert torch.allclose(out, out_ref, atol=1e-6)
torch.testing.assert_close(
    out, out_ref,
    rtol=0, atol=1e-6,
    msg = f"out should be equal to out_ref."
)
print(f"OK")