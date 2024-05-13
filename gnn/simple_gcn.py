from typing import TYPE_CHECKING
import graphblas as gb
import numpy as np
if TYPE_CHECKING:
    from gnn._typing import _GraphblasModule as gb


def convolve_features(adj: "gb.Matrix", X: np.ndarray, k: int) -> np.ndarray:
    X_hat = []
    for x in X.T:
        x_hat = gb.Vector.from_dense(x, dtype=X.dtype)
        for _ in range(k):
            x_hat << adj @ x_hat

        X_hat.append(x_hat.to_dense(dtype=X.dtype))    
    return np.array(X_hat, dtype=X.dtype).T

