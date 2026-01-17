
# Question: Implement NMS using the IoU above. Given boxes: [N,4], scores: [N], return kept indices in descending score order.
import torch

def box_iou_2d(boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float=1e-12) -> torch.Tensor:
    xA = torch.maximum(boxes1[:, None, 0], boxes2[None, :, 0]) #[N, M]
    yA = torch.maximum(boxes1[:, None, 1], boxes2[None, :, 1]) #[N, M]
    xB = torch.minimum(boxes1[:, None, 2], boxes2[None, :, 2]) #[N, M]
    yB = torch.minimum(boxes1[:, None, 3], boxes2[None, :, 3]) #[N, M]

    inter = (xB-xA).clamp_min(0.0)*(yB-yA).clamp_min(0.0) #[N, M]
    area1 = (((boxes1[:, 2] - boxes1[:, 0]).clamp_min(0.0))*((boxes1[:, 3] - boxes1[:, 1]).clamp_min(0.0)))[:, None] #[N, 1]
    area2 = (((boxes2[:, 2] - boxes2[:, 0]).clamp_min(0.0))*((boxes2[:, 3] - boxes2[:, 1]).clamp_min(0.0)))[None, :] #[1, M]
    union = area1 + area2 - inter
    return inter/union.clamp_min(eps)

def nms_axis_aligned(boxes: torch.Tensor, scores: torch.Tensor, iou_thresh: float = 0.5) -> torch.Tensor:
    # boxes: [N, 4]
    # scores: [N]
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long)
    
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break

        rest = order[1:]
        ious = box_iou_2d(boxes[i:i+1], boxes[rest]).squeeze(0) #[M]
        order = rest[ious <= iou_thresh]

    return torch.tensor(keep, dtype=torch.long)

# quick check
boxes = torch.tensor([
    [0.0, 0.0, 1.0, 1.0],
    [0.1, 0.1, 1.1, 1.1],
    [2.0, 2.0, 3.0, 3.0],
    [0.0, 0.0, 0.9, 0.9],
])
scores = torch.tensor([0.9, 0.8, 0.7, 0.85])
keep = nms_axis_aligned(boxes, scores, iou_thresh=0.5)
assert keep.ndim == 1
assert keep.dtype == torch.long
assert len(set(keep.tolist())) == keep.numel()

# reference check vs naive NMS
def nms_naive(boxes, scores, thr):
    order = scores.argsort(descending=True).tolist()
    keep = []
    while order:
        i = order.pop(0)
        keep.append(i)
        new_order = []
        for j in order:
            iou = box_iou_2d(boxes[i:i+1], boxes[j:j+1]).item()
            if iou <= thr:
                new_order.append(j)
        order = new_order

    return torch.tensor(keep, dtype=torch.long)

keep_ref = nms_naive(boxes, scores, 0.5)
assert torch.equal(keep, keep_ref)
assert torch.allclose(keep, keep_ref, atol=1e-5)
torch.testing.assert_close(
    keep, keep_ref,
    rtol=0, atol=1e-5,
    msg=f"should be same as keep_ref"
)
print(f"OK")