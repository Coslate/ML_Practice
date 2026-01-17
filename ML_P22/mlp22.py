
#Question
#Implement PCA and do reference check with pytorch PCA

import torch

def fix_signed(V: torch.Tensor) -> torch.Tensor:
    '''
    Make eigenvectors deterministic up to sign
    For each column, force the entry with largest abs value to be positive.
    '''
    # V: [D, k]
    idx = V.abs().argmax(dim=0) #[k]
    signs = torch.sign(V[idx, torch.arange(V.shape[1], device=V.device)]) #[k]
    signs = torch.where(signs==0, torch.ones_like(signs), signs) #[k], replace 0 with sign = 1
    return V*signs #[D, k]

def PCA_my(Z: torch.Tensor, k: int) -> torch.Tensor:
    Z = Z - Z.mean(dim=0, keepdim=True) #[N, D]
    cov = Z.T @ Z #[D, D]
    eigvalue, eigvector = torch.linalg.eigh(cov) #[D, D], eigenvector is column vector
    order = torch.topk(eigvalue, k=k, dim=0, largest=True, sorted=True).indices #[k], do not support complex numbers
    #order = torch.argsort(eigvalue, descending=True)[:k] #[k]
    eig_sel = fix_signed(eigvector[:, order]) #[D, k]
    return Z @ eig_sel, eig_sel #[N, k], #[D, k]

# quick check
torch.manual_seed(0)
X = torch.randn(20, 5)
Z, eig_sel = PCA_my(X, k=3)
assert Z.shape == (20, 3)

# reference check vs SVD-based PCA
k = 3
Xc = X - X.mean(dim=0, keepdim=True) #[N, D]
U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
comps_ref = fix_signed(Vh[:k].T) #[D, k]
Z_ref = Xc @ comps_ref #[N, k]

assert torch.allclose(Z, Z_ref, atol=1e-5), f"{Z} and {Z_ref} should be near."
assert torch.allclose(eig_sel, comps_ref, atol=1e-5), f"comps should be near."

# reference check vs torch.pca_kowrank
# Increasing niter makes torch.pca_lowrank’s randomized approximation converge toward the exact top-k PCA subspace, 
# so the projector error shrinks (to ~1e-5 in your case) and the check passes.
U2, S2, V2 = torch.pca_lowrank(Xc, q=k, center=False, niter=20) #V2:[D, k]
Z2 = Xc @ V2

eps = 1e-12
# using cov = X^T X vs. cov = X^T X /(N-1) won’t matter when comparing
#P_my = (Z/(Z.norm(dim=0, keepdim=True) + eps))
#P_ref = (Z2/(Z2.norm(dim=0, keepdim=True) + eps)) 
P_my = eig_sel @ eig_sel.T      # [D, D]
P_lr = V2 @ V2.T                  # [D, D]
assert torch.allclose(P_my, P_lr, atol=1e-5, rtol=0)
print(f"OK")
