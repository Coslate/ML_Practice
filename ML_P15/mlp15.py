
#Question: Given a tensor, report the first NaN/Inf index (row-major), plus basic stats.
import torch

def find_first_violation(x: torch.Tensor):
    # Returns: (idx_tuple_or_None, stats_dict)
    finite = torch.isfinite(x)
    bad = ~finite

    stats = {
        "shape": tuple(x.shape),
        "dtype": str(x.dtype),
        "device": str(x.device),
        "finite_ratio": float(finite.float().mean().item()) if x.numel() > 0 else 1.0
    }

    if x.numel() == 0:
        stats.update({"min": None, "max": None, "mean": None, "std": None})

    x_fin = x[finite]
    if x_fin.numel() > 0:
        stats.update({
            "min": float(x_fin.min().item()),
            "max": float(x_fin.max().item()),
            "mean": float(x_fin.mean().item()),
            "std": float(x_fin.std(unbiased=False).item()),
        })
    else:
        stats.update({"min": None, "max": None, "mean": None, "std": None})

    if not bad.any():
        return None, stats

    first_flat = bad.flatten().nonzero(as_tuple=False)[0, 0].item()
    idx = torch.unravel_index(torch.tensor(first_flat, device=x.device), x.shape)
    idx = tuple(int(i.item()) for i in idx)
    return idx, stats

# quick check
t = torch.tensor([
    [1.0, 2.0],
    [float('inf'), 3.0]
])
idx, stats = find_first_violation(t)
assert idx == (1, 0)
assert stats["finite_ratio"] < 1.0
assert torch.allclose(torch.tensor(stats["finite_ratio"]), torch.tensor(0.75), atol=1e-6)

t2 = torch.randn(3, 3)
idx2, stats2 = find_first_violation(t2)
assert idx2 is None
assert stats2["finite_ratio"] == 1.0

t3 = torch.tensor([float('nan'), 1.0, 2.0])
idx3, stats3 = find_first_violation(t3)
assert idx3 == (0,)
assert stats3["finite_ratio"] < 1.0
assert torch.allclose(torch.tensor(stats3["finite_ratio"]), torch.tensor(2.0/3.0), atol=1e-5)
print(f"OK")
