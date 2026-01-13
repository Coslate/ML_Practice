
# Question: Given ragged sequences (list of tensors [Ti, D]), create padded tensor padded: [B, Tmax, D] and mask: [B, Tmax].
import torch
from typing import List

def pad_and_mask(seqs: List[torch.Tensor], padding_value: float = 0.0):
    #seqs: list of tensors, each [Ti, D]

    B = len(seqs)
    assert B > 0
    D = seqs[0].shape[1]
    Tmax = max(s.shape[0] for s in seqs)

    padded = torch.full((B, Tmax, D), fill_value=padding_value, dtype=seqs[0].dtype, device=seqs[0].device)
    mask = torch.zeros((B, Tmax), dtype=torch.bool, device=seqs[0].device)

    for b, s, in enumerate(seqs):
        Ti = s.shape[0]
        padded[b, :Ti] = s
        mask[b, :Ti] = True

    return padded, mask

# quick check
torch.manual_seed(0)
seqs = [torch.randn(2, 4), torch.randn(5, 4), torch.randn(1, 4)]
padded, mask = pad_and_mask(seqs, padding_value=0.0)
assert padded.shape == (3, 5, 4)
assert mask.shape == (3, 5)
assert mask.dtype == torch.bool

# reference check
assert torch.allclose(padded[0, :2], seqs[0])
assert torch.allclose(padded[1, :5], seqs[1])
assert torch.allclose(padded[2, :1], seqs[2])
#print(f"padded[0, 2:] = {padded[0, 2:]}")
#print(f"padded[0, 2:] == 0.0 = {padded[0, 2:] == 0.0}")
assert torch.all(padded[0, 2:] == 0.0)
assert torch.all(padded[1, 5:] == 0.0)
assert torch.all(padded[2, 1:] == 0.0)
assert torch.all(mask[0, :2]) and not torch.any(mask[0, 2:])
assert torch.all(mask[1, :5]) and not torch.any(mask[1, 5:])
assert torch.all(mask[2, :1]) and not torch.any(mask[2, 1:])
print(f"OK")