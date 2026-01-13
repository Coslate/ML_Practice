
# Question:
#   Given scores: [N, C] (higher = more confident) and targets: [N, C] in {0,1},
#   compute AP per class and mAP (mean over classes with at least one positive),
#   using vectorized ops (no Python loops over N).
#
# Notes:
#   This AP is the common "ranking AP": AP_c = mean_k precision@k over all k where target==1.
#   You rank all samples by scores[:, c].
#   At any cutoff k (top-k), you treat the top-𝑘 samples as predicted positives for class c.
#   (Equivalent to area under PR curve for a ranked list in this discrete setting.)
#   Used in multi-label classification, retrieval/re-ranking, per-class ranking
#   After ranking all samples by scores[:, c], whether the true positives (targets[:, c] = 1) are concentrated near the top of the list.
#   1. If most positives are ranked near the top, AP is close to 1.
#   2. If the ranking is close to random, AP is approximately the positive portions for that class.
#   3. If the model ranks positives near the bottom, AP is close to 0.
#   Then, mAP is the mean of AP over classes that have at least one positive.
#   We don’t use an envelope because the current AP definition doesn’t include interpolation; adding it would compute a different metric.
#   APc = (1/Pc)*(Sum(Precision@k when t[k, c] == 1, for top-k)), it is still the area under PR curve, the recall here is 1/Pc, 
#   only when t[k, c] == 1 will increase the recall, precision, and the area under PR-curve, so APc only calculates those with t[k, c] == 1.

import torch

def map_ap(scores: torch.Tensor, targets: torch.Tensor, eps: float=1e-12):
    # scores: [N, C] float
    # targets: [N, C] {0, 1} or bool
    # returns: mAP scalar tensor, ap_per_class [C]

    assert scores.ndim == 2 and targets.ndim == 2
    N, C = scores.shape
    t = targets.to(dtype=torch.float32)

    #sort each class by score desc: order [N, C]
    order = scores.argsort(dim=0, descending=True)
    t_sorted = t.gather(dim=0, index=order)
    tp = t_sorted.cumsum(dim=0) #[N, C]
    fp = (1.0-t_sorted).cumsum(dim=0) #[N, C]

    precision = tp/(tp+fp).clamp_min(eps) #[N, C]
    pos = t.sum(dim=0) #[C]
    ap = (precision*t_sorted).sum(dim=0) / pos.clamp_min(eps) #[C]

    valid = pos > 0
    mAP = ap[valid].mean() if valid.any() else ap.new_tensor(0.0) # scalar
    return mAP, ap


# quick check
torch.manual_seed(0)
N, C = 20, 5
scores = torch.randn(N, C) #Normal Dist. range is unbounded, mean=0, var=1
targets = (torch.rand(N, C)> 0.7) # sparse positives, Uniform Dist. [0, 1]
mAP, ap = map_ap(scores, targets)
assert ap.shape == (C, )
assert torch.isfinite(ap).all()
assert torch.isfinite(mAP).all()
assert 0.0 <= mAP.item() <= 1.0 + 1e-6

# reference check vs naive
def ap_naive(scores_1c: torch.Tensor, targets_1c: torch.Tensor, eps: float=1e-12) -> torch.Tensor:
    # scores_1c: [N]
    # targets_1c: [N] in {0, 1} or bool

    t = targets_1c.to(torch.float32)
    pos = t.sum().item()
    if pos == 0:
        return scores_1c.new_tensor(0.0)

    order = torch.argsort(scores_1c, descending=True)
    t_sorted = t[order]

    tp = 0.0
    ap_sum = 0.0
    for k in range(t_sorted.numel()):
        if t_sorted[k].item() == 1.0:
            tp += 1.0
            ap_sum += tp/(k+1.0)

    return scores_1c.new_tensor(ap_sum/(pos+eps))

ap_ref = torch.stack([ap_naive(scores[:, c], targets[:, c]) for c in range(C)], dim=0) #[C]
assert torch.allclose(ap, ap_ref, atol=1e-6, rtol=1e-6)

valid = targets.sum(dim=0) > 0 #[C]
mAP_ref = ap_ref[valid].mean() if valid.any() else ap_ref.new_tensor(0.0)
assert torch.allclose(mAP, mAP_ref, atol=1e-6, rtol=1e-6)
print(f"OK")

