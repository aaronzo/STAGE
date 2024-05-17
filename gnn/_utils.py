import graphblas as gb
from typing import TYPE_CHECKING, Optional, Union, Callable
import functools as ft
import torch
import numpy as np
from torch_sparse import SparseTensor
if TYPE_CHECKING:
    from gnn._typing import _GraphblasModule as gb
    from graphblas.core.matrix import MatrixExpression, TransposedMatrix
    from numpy.typing import DTypeLike


def eval_expr(
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
    if isinstance(edge_index, SparseTensor):
        return torch_sparse_tensor_to_graphblas(edge_index, weighted=weighted, dtype=dtype)
    if edge_index.is_sparse_csr:
        return torch_sparse_csr_to_graphblas(edge_index, weighted=weighted, dtype=dtype)
    return torch_edge_index_to_graphblas(edge_index, num_nodes=num_nodes, dtype=dtype)


def torch_sparse_csr_to_graphblas(
    adj_t: torch.Tensor,
    *,
    weighted: bool = False,
    dtype: "Optional[DTypeLike]" = None
) -> gb.Matrix:
    return _torch_sparse_csr_to_graphblas(adj_t, weighted, dtype)


def torch_sparse_tensor_to_graphblas(
    adj_t: SparseTensor,
    *,
    weighted: bool = False,
    dtype: "Optional[DTypeLike]" = None
) -> gb.Matrix:
    return torch_sparse_csr_to_graphblas(
        adj_t.to_torch_sparse_csr_tensor(),
        weighted=weighted,
        dtype=dtype,
    )


def torch_edge_index_to_graphblas(
    edge_index: Union[torch.Tensor, SparseTensor],
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


def transpose_if(
    condition: bool,
    matrix: "Union[gb.Matrix, MatrixExpression]",
) -> "Union[MatrixExpression, MatrixExpression, TransposedMatrix]":
    return matrix.T if condition else matrix


# jit compile with the correct types, during call of triangle
def _recover_triad_count(x):
    return (1 - x) / 2 if x % 2 else x


recover_triad_count: Callable[[gb.Matrix], "MatrixExpression"] = (
    gb.unary.register_anonymous(_recover_triad_count)
)