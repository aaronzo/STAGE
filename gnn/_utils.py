import graphblas as gb
from typing import TYPE_CHECKING, Optional
import functools as ft
import torch
if TYPE_CHECKING:
    from gnn._typing import _GraphblasModule as gb
    from graphblas.core.matrix import MatrixExpression
    from numpy.typing import DTypeLike


def eval(
    expr: "MatrixExpression",
    out: Optional[gb.Matrix] = None,
    **kw
) -> gb.Matrix:
    if out is None:
        return expr.new(**kw)
    out(**kw) << expr
    return out


def torch_to_graphblas(
    edge_index: torch.Tensor,
    *,
    num_nodes: Optional[int] = None,
    weighted: bool = False,
    dtype: "Optional[DTypeLike]" = None,
) -> gb.Matrix:
    if edge_index.is_sparse_csr:
        import sys
        print("IS SPARSE", file=sys.stderr)
        return torch_sparse_csr_to_graphblas(edge_index, weighted=weighted, dtype=dtype)
    return torch_edge_index_to_graphblas(edge_index, num_nodes=num_nodes, dtype=dtype)


def torch_sparse_csr_to_graphblas(
    adj_t: "torch.Tensor",
    *,
    weighted: bool = False,
    dtype: "Optional[DTypeLike]" = None
) -> gb.Matrix:
    return _torch_sparse_csr_to_graphblas(adj_t, weighted, dtype)


def torch_edge_index_to_graphblas(
    edge_index: torch.Tensor,
    *,
    num_nodes: Optional[int] = None,
    dtype: "Optional[DTypeLike]" = None,
) -> gb.Matrix:
    return _torch_edge_index_to_graphblas(edge_index, num_nodes, dtype)


@ft.lru_cache(maxsize=1)
def _torch_sparse_csr_to_graphblas(
    adj_t: torch.Tensor,
    weighted: bool,
    dtype: "Optional[DTypeLike]",
) -> gb.Matrix:
    if not adj_t.is_sparse_csr:
        adj_t = adj_t.to_sparse_csr()
    return gb.Matrix.from_csr(
        indptr=adj_t.crow_indices().detach().cpu().numpy(),
        col_indices=adj_t.col_indices().detach().cpu().numpy(),
        values=1.0 if not weighted else adj_t.values().detach().cpu().numpy(),
        nrows=adj_t.shape[0],
        ncols=adj_t.shape[0],
        dtype=dtype,
    )

@ft.lru_cache(maxsize=1)
def _torch_edge_index_to_graphblas(
    edge_index: torch.Tensor,
    num_nodes: Optional[int],
    dtype: "Optional[DTypeLike]",
) -> gb.Matrix:
    return gb.Matrix.from_coo(*edge_index, dtype=dtype, nrows=num_nodes, ncols=num_nodes)
