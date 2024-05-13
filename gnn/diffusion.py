"""Graph Diffusion operators which can be pre-computed."""

import numpy as np
from typing import TYPE_CHECKING, Iterator, Callable, Union, Literal, Sequence
import graphblas as gb
from pathlib import Path
import itertools
from gnn import _utils

from datasets import Dataset

if TYPE_CHECKING:
    from gnn._typing import _GraphblasModule as gb


def rw_norm(adj: gb.Matrix, *, copy: bool = True) -> gb.Matrix:
    return _utils.eval(adj / adj.reduce_rowwise(), out=None if copy else adj)


def gcn_norm(adj: gb.Matrix, *, copy: bool = True) -> gb.Matrix:
    sqrt_deg = adj.reduce_rowwise().apply(gb.unary.sqrt)
    return _utils.eval(sqrt_deg * adj * sqrt_deg, out=None if copy else adj)


def norm(adj: gb.Matrix, method: Union[Literal["gcn"], Literal["rw"]], *, copy: bool = True) -> gb.Matrix:
    if method == "rw":
        return rw_norm(adj, copy=copy)
    return gcn_norm(adj, copy=copy)


def simple(adj: gb.Matrix) -> Callable[[np.ndarray], np.ndarray]:
    def diffuse_fn(X: np.ndarray) -> np.ndarray:
        X_hat = []
        for x in X.T:
            x_hat = gb.Vector.from_dense(x, dtype=X.dtype)
            x_hat << adj @ x_hat
            X_hat.append(x_hat.to_dense(dtype=X.dtype))    
        return np.array(X_hat, dtype=X.dtype).T
    return diffuse_fn


def power(adj: gb.Matrix, k: int) -> Callable[[np.ndarray], np.ndarray]:
    def diffuse_fn(X: np.ndarray) -> np.ndarray:
        X_hat = []
        for x in X.T:
            x_hat = gb.Vector.from_dense(x, dtype=X.dtype)
            for _ in range(k):
                x_hat << adj @ x_hat
            X_hat.append(x_hat.to_dense(dtype=X.dtype))    
        return np.array(X_hat, dtype=X.dtype).T
    return diffuse_fn


def appnp(adj: gb.Matrix, alpha: float = 0.15, iterations: int = 50) -> np.ndarray:
    beta = 1 - alpha
    def diffuse_fn(X: np.ndarray) -> np.ndarray:
        X_hat = []
        for x in X.T:
            x = gb.Vector.from_dense(x, dtype=X.dtype)
            x_hat = x.dup()
            for _ in range(iterations):
                x_hat << beta * (adj @ x_hat) + alpha * x
            
            X_hat.append(x_hat.to_dense(dtype=X.dtype))
        return np.array(X_hat).T
    return diffuse_fn


def triangle(adj: gb.Matrix, directed: bool = True) -> Callable[[np.ndarray], np.ndarray]:
    A = gb.select.offdiag(adj).S.new(dtype=adj.dtype)
    B = gb.select.offdiag(A @ A).new(dtype=adj.dtype)

    mask = gb.unary.identity(A.T).S if directed else A.S
    A(mask=mask) << B
    
    return simple(A)


def diffuse_powers(diffuse: Callable[[np.ndarray], np.ndarray], X: np.ndarray,  k: int) -> Iterator[np.ndarray]:
    for _ in range(k):
        X = diffuse(X)
        yield X


class SimpleGCNDiffusion:
    def __init__(self, k, path: Path) -> None:
        self.k = k
        self.path = path
    
    def propagate(self, adj: gb.Matrix, X: np.ndarray) -> np.ndarray:
        adj_gcn = gcn_norm(adj)
        return power(adj_gcn, self.k)(X)


class SIGNDiffusion:
    def __init__(self, s: int, p: int, t: int, norms: Sequence[str] =("gcn", "rw", "rw")) -> None:
        self.s = s
        self.p = p
        self.t = t
        self._norms = norms
    
    @property
    def r(self) -> int:
        return self.s + self.p + self.t
    
    def propagate(self, adj: gb.Matrix, X: np.ndarray) -> np.ndarray:
        normed_adjs = {m: norm(adj, method=m) for m in set(self._norms[:2])}
        embeddings_iter = itertools.chain(
            diffuse_powers(simple(normed_adjs[0]), X, self.s),
            diffuse_powers(appnp(normed_adjs[1]), X, self.p),
            diffuse_powers(norm(triangle(adj), method=self._norms[2], copy=False), X, self.t)
        )
        return np.hstack(tuple(embeddings_iter), dtype=X.dtype)

        Dataset.from_n