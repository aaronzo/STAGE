"""Graph Diffusion operators which can be pre-computed."""

import numpy as np
from typing import TYPE_CHECKING, Iterator, Callable, Union, Literal, Tuple
import graphblas as gb
from itertools import chain
from gnn import _utils
import functools as ft
from abc import ABCMeta, abstractmethod
from pathlib import Path
import torch

if TYPE_CHECKING:
    from gnn._typing import _GraphblasModule as gb


@ft.lru_cache(maxsize=1)
def rw_norm(adj: gb.Matrix, *, copy: bool = True) -> gb.Matrix:
    return _utils.eval(adj / adj.reduce_rowwise(), out=None if copy else adj)


@ft.lru_cache(maxsize=1)
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
            X_hat.append(x_hat.to_dense(dtype=X.dtype, fill_value=0))
        return np.array(X_hat, dtype=X.dtype).T
    return diffuse_fn


def power(adj: gb.Matrix, k: int) -> Callable[[np.ndarray], np.ndarray]:
    def diffuse_fn(X: np.ndarray) -> np.ndarray:
        X_hat = []
        for x in X.T:
            x_hat = gb.Vector.from_dense(x, dtype=X.dtype)
            for _ in range(k):
                x_hat << adj @ x_hat
            X_hat.append(x_hat.to_dense(dtype=X.dtype, fill_value=0))    
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


def triangle(adj: gb.Matrix, directed: bool = True) -> gb.Matrix:
    A = gb.select.offdiag(adj).S.new(dtype=adj.dtype)
    mask = gb.unary.identity(A.T).S if directed else A.S
    A(mask=mask) << A @ A
    return A


def diffuse_powers(diffuse: Callable[[np.ndarray], np.ndarray], X: np.ndarray,  k: int) -> Iterator[np.ndarray]:
    for _ in range(k):
        X = diffuse(X)
        yield X


class Diffusion(metaclass=ABCMeta):
    @abstractmethod
    def num_features(self, in_features: int) -> int:
        ...

    @abstractmethod
    def propagate(self, adj: gb.Matrix, X: np.ndarray) -> np.ndarray:
        ...

    def propagate_torch(
        self,
        edge_index: torch.Tensor,
        X: torch.Tensor,
        cache_location: Union[str, Path, None] = None,
    ) -> torch.Tensor:
        shape = (X.shape[0], self.num_features(X.shape[1]))
        if cache_location is not None:
            path = Path(cache_location)
            if path.exists() and path.is_file():
                return torch.from_numpy(self._load_emb(path, shape=shape)).to(X.dtype)
        
        adj = _utils.torch_to_graphblas(edge_index, num_nodes=X.shape[0])
        X_np = X.detach().cpu().numpy()
        out = self.propagate(adj, X_np)
        if cache_location is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._save_emb(out, cache_location, shape)
        return torch.from_numpy(out).to(device=X.device)

    def _load_emb(self, path: Path, shape: Tuple[int, int]) -> np.ndarray:
        return np.array(np.memmap(
            path,
            mode='r',
            dtype=np.float16,
            shape=shape,
        ))
    
    def _save_emb(self, X: np.ndarray, path: Path, shape: Tuple[int, int]) -> None:
        mm = np.memmap(
            path,
            dtype=np.float16,
            mode='w+',
            shape=shape
        )
        mm[:] = X[:]

class SimpleGCNDiffusion(Diffusion):
    def __init__(self, k,) -> None:
        self.k = k
    
    def num_features(self, in_features: int) -> int:
        return in_features

    def propagate(self, adj: gb.Matrix, X: np.ndarray) -> np.ndarray:
        adj_gcn = gcn_norm(adj)
        return power(adj_gcn, self.k)(X)


class SIGNDiffusion(Diffusion):
    def __init__(
        self, 
        s: int,
        p: int,
        t: int,
        s_norm: Union[Literal["gcn"], Literal["rw"]] = "gcn",
        p_norm: Union[Literal["gcn"], Literal["rw"]] = "rw",
        t_norm: Union[Literal["gcn"], Literal["rw"]] = "rw",
    ) -> None:
        self.s = s
        self.p = p
        self.t = t
        self.s_norm = s_norm
        self.p_norm = p_norm
        self.t_norm = t_norm
        if self.r < 1:
            raise ValueError
    
    @property
    def r(self) -> int:
        return self.s + self.p + self.t
    
    def num_features(self, in_features: int) -> int:
        return self.r * in_features

    def propagate(self, adj: gb.Matrix, X: np.ndarray) -> np.ndarray:
        ops = []
        print("PROPAGATING SIGN")
        if (s := self.s):
            simple_diffuser = simple(norm(adj, method=self.s_norm))
            ops.append(diffuse_powers(simple_diffuser, X, s))
        if (p := self.p):
            ppr_diffuser = appnp(norm(adj, method=self.p_norm))
            ops.append(diffuse_powers(ppr_diffuser, X, p))
        if (t := self.t):
            adj_triangle = triangle(adj, directed=False)
            triangle_diffuser = simple(norm(adj_triangle, method=self.t_norm, copy=False))
            ops.append(diffuse_powers(triangle_diffuser, X, t))

        return np.hstack(tuple(chain(*ops)), dtype=X.dtype)
