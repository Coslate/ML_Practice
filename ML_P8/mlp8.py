
# Question: Given A: [B, N, D] and B_: [B, M, D], compute dots: [B, N, M] where dots[b,i,j] = A[b,i]·B_[b,j].
import torch

def batched_dot(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    #A: [B, N, D]
    #B: [B, M, D]
    #return A @ B.permute(0, 2, 1)
    return torch.bmm(A, B.permute(0, 2, 1)) #[B, N, M]

#quick check
torch.manual_seed(0)
A = torch.randn(2, 4, 3)
B_ = torch.randn(2, 5, 3)
dots = batched_dot(A, B_)
assert dots.shape == (2, 4, 5)

#reference check vs naive
dots_ref = torch.zeros((2, 4, 5))
for b in range(2):
    for i in range(4):
        for j in range(5):
            dots_ref[b, i, j] = (A[b, i]*B_[b, j]).sum()

assert torch.allclose(dots, dots_ref, atol=1e-5)
print(f"OK")
