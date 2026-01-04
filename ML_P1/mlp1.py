

#Question: Given X: [N, D], compute dist2: [N, N] where dist2[i,j] = ||X[i]-X[j]||^2, fully vectorized.

import torch

def pairwise_disst2(X: torch.Tensor) -> torch.Tensor:
    #X: [N, D]
    #dist2: [N, N]
    diff = X[:, None, :] - X[None, :, :] #[N, 1, D] - [1, N, D] = [N, N, D]
    dist2 = (diff*diff).sum(dim=-1) #[N, N]
    return dist2

# quick check
torch.manual_seed(0)
X = torch.randn(5, 3)
d2 = pairwise_disst2(X)
#assert d2.shape  == (5, 5)
#assert torch.allclose(torch.diag(d2), torch.zeros(5), atol=1e-6)
torch.testing.assert_close(
    d2.shape, (5, 5),
    rtol=0, atol=1e-8,
    msg=f"d2.shape should be (5, 5), but got {d2.shape}."
)
torch.testing.assert_close(
    torch.diag(d2), torch.zeros(5),
    rtol=0.0, atol=1e-6,
    msg="diag(d2) should be all zeros, but values differ."
)

# reference check vs naive
d2_ref = torch.zeros(5, 5)
for i in range(5):
    for j in range(5):
        d2_ref[i, j] = ((X[i] - X[j])**2).sum()

#assert torch.allclose(d2, d2_ref, atol=1e-6)
torch.testing.assert_close(
    d2, d2_ref,
    rtol=0, atol=1e-6,
    msg = f"d2 value differs with d2_ref."
)
print(f"OK")
