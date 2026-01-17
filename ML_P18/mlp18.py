
import torch
from typing import Dict, Set, List, Optional, Tuple
#run by: python -m pytest -q ./mlp18.py


# ------------------------------------------------------------
# Optional helper: build a topological order over PyTorch autograd
# Function nodes (similar to your find_topo_sort, but on grad_fn graph).
# ------------------------------------------------------------
def find_autograd_topo(output_tensor: torch.Tensor) -> List[object]:
    """
    Return a topological order (inputs-first) of autograd Function nodes
    reachable from output_tensor.grad_fn. If output_tensor is a leaf,
    returns [].

    NOTE: This is for inspection/teaching. PyTorch itself schedules the
    reverse-topo traversal internally during backward().
    """
    fn0 = output_tensor.grad_fn
    if fn0 is None:
        return []

    visited: Set[int] = set()
    topo: List[object] = []

    def dfs(fn: object) -> None:
        fid = id(fn)
        if fid in visited:
            return
        visited.add(fid)

        # Traverse predecessors (toward inputs)
        # Each entry is (next_fn, input_nr) where next_fn can be None
        for next_fn, _ in getattr(fn, "next_functions", []):
            if next_fn is not None:
                dfs(next_fn)

        topo.append(fn)

    dfs(fn0)
    return topo


def _collect_leaf_tensors(output_tensor: torch.Tensor) -> Set[torch.Tensor]:
    """
    Collect leaf tensors (requires_grad=True, grad_fn=None) reachable from output_tensor.
    Includes output_tensor itself if it is a leaf and requires_grad=True.
    """
    leaves: Set[torch.Tensor] = set()

    # Special-case: output itself is a leaf
    if output_tensor.grad_fn is None:
        if output_tensor.requires_grad:
            leaves.add(output_tensor)
        return leaves

    def walk(fn: object) -> None:
        if fn is None:
            return

        # Leaf tensors are represented by AccumulateGrad nodes in the autograd graph
        if fn.__class__.__name__ == "AccumulateGrad":
            # AccumulateGrad has attribute `.variable` pointing to the leaf tensor
            v = fn.variable
            if v.requires_grad and v.grad_fn is None:
                leaves.add(v)
            return

        for next_fn, _ in getattr(fn, "next_functions", []):
            if next_fn is not None:
                walk(next_fn)

    walk(output_tensor.grad_fn)
    return leaves


# ------------------------------------------------------------
# Main function: PyTorch version of compute_gradient_of_variables
# ------------------------------------------------------------
def compute_gradient_of_variables_torch(
    output_tensor: torch.Tensor,
    out_grad: Optional[torch.Tensor] = None,
    *,
    retain_graph: bool = False,
    create_graph: bool = False,
    zero_existing_grads: bool = True,
    return_topo: bool = False,
) -> Dict[torch.Tensor, torch.Tensor] | Tuple[Dict[torch.Tensor, torch.Tensor], List[object]]:
    """
    PyTorch analogue of your compute_gradient_of_variables (Needle):

    - Treat output_tensor as the output node.
    - Seed gradient with out_grad (defaults to ones_like(output_tensor)).
    - Collect all leaf tensors reachable from output_tensor.
    - Run output_tensor.backward(out_grad, ...) so PyTorch performs reverse-topo traversal internally.
    - Return {leaf_tensor: leaf_tensor.grad}.

    If return_topo=True, also returns find_autograd_topo(output_tensor) (inputs-first Function list).

    Notes:
      * If output_tensor is non-scalar, out_grad must have the same shape.
      * Intermediate tensor grads are not stored unless you call .retain_grad() on them.
      * Gradients on leaves accumulate unless you clear them (zero_existing_grads=True).
    """
    if out_grad is None:
        out_grad = torch.ones_like(output_tensor)

    if out_grad.shape != output_tensor.shape:
        raise ValueError(
            f"out_grad.shape {out_grad.shape} must match output_tensor.shape {output_tensor.shape}"
        )

    leaves = _collect_leaf_tensors(output_tensor)

    if zero_existing_grads:
        for v in leaves:
            v.grad = None  # like optimizer.zero_grad(set_to_none=True)

    # PyTorch engine does reverse-topological traversal + per-op backward internally
    output_tensor.backward(out_grad, retain_graph=retain_graph, create_graph=create_graph)

    grads: Dict[torch.Tensor, torch.Tensor] = {v: v.grad for v in leaves}

    if return_topo:
        topo = find_autograd_topo(output_tensor)
        return grads, topo
    return grads


# ------------------------------------------------------------
# Unit tests (pytest)
# Save this file and run: pytest -q
# ------------------------------------------------------------
def _allclose(a: torch.Tensor, b: torch.Tensor, atol=1e-6, rtol=1e-6) -> bool:
    return torch.allclose(a, b, atol=atol, rtol=rtol)

def test_scalar_output_matches_autograd_grad():
    torch.manual_seed(0)
    x = torch.randn(7, requires_grad=True)
    y = torch.randn(7, requires_grad=True)

    # 1) reference grads (consumes the graph)
    out1 = (x * y + x.square()).sum()
    gx_ref, gy_ref = torch.autograd.grad(out1, [x, y])

    # 2) recompute out to build a fresh graph, then test your function
    out2 = (x * y + x.square()).sum()
    grads = compute_gradient_of_variables_torch(out2)
    gx = grads[x]
    gy = grads[y]

    assert torch.allclose(gx, gx_ref, atol=1e-6, rtol=1e-6)
    assert torch.allclose(gy, gy_ref, atol=1e-6, rtol=1e-6)


def test_non_scalar_output_with_out_grad_matches_autograd_grad():
    torch.manual_seed(1)
    x = torch.randn(4, 3, requires_grad=True)
    W = torch.randn(3, 5, requires_grad=True)

    out_grad = torch.randn(4, 5)

    # 1) reference grads (consumes the graph)
    out1 = x @ W
    gx_ref, gW_ref = torch.autograd.grad(out1, [x, W], grad_outputs=out_grad)

    # 2) recompute out to build a fresh graph, then test your function
    out2 = x @ W
    grads = compute_gradient_of_variables_torch(out2, out_grad)
    gx = grads[x]
    gW = grads[W]

    assert torch.allclose(gx, gx_ref, atol=1e-6, rtol=1e-6)
    assert torch.allclose(gW, gW_ref, atol=1e-6, rtol=1e-6)


def test_branching_accumulation_is_correct():
    torch.manual_seed(2)
    x = torch.randn(6, requires_grad=True)
    y = torch.randn(6, requires_grad=True)

    out = (x * y).sum() + (x * y).sum()  # exactly 2 * sum(x*y)
    grads = compute_gradient_of_variables_torch(out)

    assert _allclose(grads[x], 2.0 * y)
    assert _allclose(grads[y], 2.0 * x)


def test_zero_existing_grads_true_does_not_accumulate():
    torch.manual_seed(3)
    x = torch.randn(5, requires_grad=True)
    y = torch.randn(5, requires_grad=True)

    out = (x * y).sum()

    grads1 = compute_gradient_of_variables_torch(out, zero_existing_grads=True, retain_graph=True)
    gx1 = grads1[x].clone()
    gy1 = grads1[y].clone()

    grads2 = compute_gradient_of_variables_torch(out, zero_existing_grads=True, retain_graph=False)
    assert _allclose(grads2[x], gx1)
    assert _allclose(grads2[y], gy1)


def test_zero_existing_grads_false_accumulates():
    torch.manual_seed(4)
    x = torch.randn(5, requires_grad=True)
    y = torch.randn(5, requires_grad=True)

    out = (x * y).sum()

    grads1 = compute_gradient_of_variables_torch(out, zero_existing_grads=True, retain_graph=True)
    gx1 = grads1[x].clone()
    gy1 = grads1[y].clone()

    grads2 = compute_gradient_of_variables_torch(out, zero_existing_grads=False, retain_graph=False)
    assert _allclose(grads2[x], 2.0 * gx1)
    assert _allclose(grads2[y], 2.0 * gy1)


def test_out_grad_shape_mismatch_raises():
    torch.manual_seed(5)
    x = torch.randn(2, 3, requires_grad=True)
    out = x.sum(dim=1)  # shape [2]

    bad_out_grad = torch.ones(2, 1)  # wrong shape
    try:
        _ = compute_gradient_of_variables_torch(out, bad_out_grad)
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_leaf_output_is_included_and_gets_grad():
    # This is the special-case patch: output_tensor itself is a leaf
    x = torch.randn(3, requires_grad=True)  # leaf
    out = x                                 # leaf output
    out_grad = torch.tensor([1.0, 2.0, 3.0])

    grads = compute_gradient_of_variables_torch(out, out_grad)
    assert x in grads
    assert _allclose(grads[x], out_grad)
    assert _allclose(x.grad, out_grad)