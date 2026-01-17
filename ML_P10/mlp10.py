
# Question: Given boxes1: [N, 4] and boxes2: [M, 4] as [x1, y1, x2, y2], compute iou: [N, M] vectorized.
import torch

def box_iou_2d(boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float=1e-12) -> torch.Tensor:
    # boxes1: [N, 4]
    # boxes2: [M, 4]
    b1 = boxes1
    b2 = boxes2

    xA = torch.maximum(b1[:, None, 0], b2[None, :, 0]) #[N, M]
    yA = torch.maximum(b1[:, None, 1], b2[None, :, 1]) #[N, M]
    xB = torch.minimum(b1[:, None, 2], b2[None, :, 2]) #[N, M]
    yB = torch.minimum(b1[:, None, 3], b2[None, :, 3]) #[N, M]

    inter = (xB-xA).clamp_min(0.0) * (yB-yA).clamp_min(0.0)
    area1 = ((b1[:, 2] - b1[:, 0]).clamp_min(0.0) * (b1[:, 3] - b1[:, 1]).clamp_min(0.0))[:, None] #[N, 1]
    area2 = ((b2[:, 2] - b2[:, 0]).clamp_min(0.0) * (b2[:, 3] - b2[:, 1]).clamp_min(0.0))[None, :] #[1, M]

    union = area1 + area2 - inter #[N, M]
    return inter/union.clamp_min(eps)

# quick check
boxes1 = torch.tensor([[0, 0, 1, 1],
                       [0, 0, 2, 2]], device='cpu', dtype=torch.float32)

boxes2 = torch.tensor([[0.5, 0.5, 1.5, 1.5],
                       [2.0, 2.0, 3.0, 3.0]], device='cpu', dtype=torch.float32)

iou = box_iou_2d(boxes1, boxes2)
assert iou.shape == (2, 2)
assert torch.isfinite(iou).all()

iou_ref = torch.zeros(2, 2)
for i in range(2):
    for j in range(2):
        xA = max(boxes1[i, 0].item(), boxes2[j, 0].item())
        yA = max(boxes1[i, 1].item(), boxes2[j, 1].item())
        xB = min(boxes1[i, 2].item(), boxes2[j, 2].item())
        yB = min(boxes1[i, 3].item(), boxes2[j, 3].item())

        inter = max(0, xB-xA) * max(0.0, yB-yA)
        a1 = max(0.0, (boxes1[i, 2] - boxes1[i, 0]).item()) * max(0.0, (boxes1[i, 3] - boxes1[i, 1]).item())
        a2 = max(0.0, (boxes2[j, 2] - boxes2[j, 0]).item()) * max(0.0, (boxes2[j, 3] - boxes2[j, 1]).item())
        union = a1+a2-inter
        iou_ref[i, j] = inter/max(union, 1e-12)

assert torch.allclose(iou, iou_ref, atol=1e-6)
torch.testing.assert_close(
    iou, iou_ref,
    rtol=0.0, atol=1e-6,
    msg=f"should be the same as iou_ref"
)

print(f"OK")