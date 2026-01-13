
# Question:
# KITTI-style AP|R40 (40 recall positions) for 3D object detection
#
# Expected inputs:
#   scores:   [N, C] detection confidence scores (higher = more confident)
#   tp_flags: [N, C] in {0,1} after IoU matching (1 = true positive, 0 = false positive)
#   num_gt:   [C] number of GT objects for each class AFTER difficulty filtering
#
# Notes:
#   - This code only covers the PR/AP integration part.
#   - You must generate tp_flags via KITTI matching rules (per image, per class,
#     1-to-1 matching with IoU threshold; ignore DontCare, etc.) before calling this.
#   - AP|R40 uses 40 recall positions r = {1/40, 2/40, ..., 40/40}.
#
# Vectorization constraint:
#   - No Python loops over N. We allow a small loop over C for searchsorted.

import torch

def kitti_map_ap_r40(
        scores: torch.Tensor,
        tp_flags: torch.Tensor,
        num_gt: torch.Tensor,
        eps: float=1e-12
):
    # scores: [N, C] float
    # tp_flags: [N, C] {0, 1} or bool (1=TP, 0=FP) after matching
    # num_gt: [C] int/float (GT count per class)
    # returns: (mAP scalar tensor, ap_per_class [C])
    assert scores.ndim == 2 and tp_flags.ndim == 2
    assert scores.shape == tp_flags.shape
    assert num_gt.ndim == 1 and num_gt.numel() == scores.shape[1]

    N, C = scores.shape
    tp = tp_flags.to(dtype=torch.float32)
    gt = num_gt.to(dtype=torch.float32)

    # 1. sort each class by score descending
    order = scores.argsort(dim=0, descending=True) #[N, C]
    tp_sorted = tp.gather(dim=0, index=order)      #[N, C], tp_sorted[i,c]=tp[order[i,c],c]

    # 2. cumulative TP/FP
    tp_cum = tp_sorted.cumsum(dim=0) #[N, C]
    fp_cum = (1.0-tp_sorted).cumsum(dim=0) #[N, C]

    # 3. precision/recall curve
    precision = tp_cum/(tp_cum+fp_cum).clamp_min(eps) #[N,C]
    recall = tp_cum/gt.clamp_min(1.0).view(1, C) #[N, C]

    # 4. prevision envelope: p_interp(r) = max_(r' >= r) p(r')
    #    Make precision non-increasing w.r.t recall by sweeping from back
    precision_env = torch.flip(
        torch.cummax(torch.flip(precision, dims=[0]), dim=0).values,
        dims=[0],
    ) #[N, C]

    # 5. sample at 40 recall positions R40 = {1/40, ..., 1}
    r = torch.arange(1, 41, device = scores.device, dtype = torch.float32)/40.0 #[40]

    ap = scores.new_zeros((C,), dtype=torch.float32)
    for c in range(C):
        if gt[c].item() <= 0:
            ap[c] = 0.0
            continue

        # recall[:, c] is non-decreasingl find first index where recall >= r_i
        idx = torch.searchsorted(recall[:, c].contiguous(), r, right=False) #[40]

        # if recall never reaches r_i, precision at that recall level is 0
        valid = idx < N
        idx_safe = idx.clamp(max=N-1)
        p_at_r = torch.where(valid, precision_env[idx_safe, c], precision_env.new_zeros(())) #[40]
        ap[c] = p_at_r.mean()

    valid_cls = gt > 0 #[C]
    mAP = ap[valid_cls].mean() if valid_cls.any() else ap.new_tensor(0.0)
    return mAP, ap

# quick check
torch.manual_seed(0)
N, C = 200, 3
scores = torch.randn(N, C)

# make synthestic GT counts
num_gt = torch.tensor([30, 10, 0], dtype=torch.float32) #last class has no GT

# make synthetic tp_flags with TP <= numgt (loops over C allowed only in test)
tp_flags = torch.zeros((N, C), dtype=torch.bool)
for c in range(C):
    if num_gt[c].item() > 0:
        k = int(num_gt[c].item()*0.7) #assume recall not perfect
        idx = torch.randperm(N)[:k]
        tp_flags[idx, c] = True

mAP, ap = kitti_map_ap_r40(scores, tp_flags, num_gt)
assert ap.shape == (C,)
assert torch.isfinite(ap).all()
assert torch.isfinite(mAP).all()
assert 0.0 <= mAP.item() <= 1.0 + 1e-6

# reference check vs naive
def ap_r40_naive(scores_1c: torch.Tensor, tp_1c: torch.Tensor, num_gt_1c: float, eps: float = 1e-12):
    if num_gt_1c <= 0:
        return scores_1c.new_tensor(0.0)

    order = torch.argsort(scores_1c, descending=True) #[N]
    tp_sorted = tp_1c.to(torch.float32)[order] #[N]
    fp_sorted = 1.0 - tp_sorted #[N]

    tp_cum = torch.cumsum(tp_sorted, dim=0) #[N]
    fp_cum = torch.cumsum(fp_sorted, dim=0) #[N]

    precision = tp_cum/(tp_cum+fp_cum).clamp_min(eps) #[N]
    recall = tp_cum/(num_gt_1c+0.0) #[N]

    # precision envelope
    prec_env = torch.flip(torch.cummax(torch.flip(precision, dims=[0]), dim=0).values, dims=[0]) #[N]

    r = torch.arange(1, 41, device=scores_1c.device, dtype=torch.float32)/40.0 #[40]
    idx = torch.searchsorted(recall.contiguous(), r, right=False) #[40]
    valid = idx < scores_1c.numel() #[40]
    idx_safe = idx.clamp(max=scores_1c.numel()-1) #[40]
    p_at_r = torch.where(valid, prec_env[idx_safe], prec_env.new_zeros(())) #[40]
    return p_at_r.mean()

ap_ref = torch.stack([ap_r40_naive(scores[:, c], tp_flags[:, c], float(num_gt[c].item())) for c in range(C)], dim=0)
assert torch.allclose(ap, ap_ref, atol=1e-6, rtol=1e-6)

valid = num_gt > 0
mAP_ref = ap_ref[valid].mean() if valid.any() else ap_ref.new_tensor(0.0)
assert torch.allclose(mAP, mAP_ref, atol=1e-6, rtol=1e-6)
print(f"OK")
