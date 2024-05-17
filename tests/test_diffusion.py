from gnn.diffusion import triangle
import numpy as np
import graphblas as gb
import networkx as nx
import pytest
from typing import TYPE_CHECKING
import math
import scipy.sparse as sp

if TYPE_CHECKING:
    from gnn._typing import _GraphblasModule as gb


def to_scipy(G: nx.Graph) -> sp.csr_array:
    return nx.adjacency_matrix(G, nodelist=sorted(G.nodes), dtype=np.float32)


def test_triangle_directed():
    expected = to_scipy(nx.cycle_graph(3, create_using=nx.DiGraph))
    adj = gb.io.from_scipy_sparse(expected)
    assert np.allclose(triangle(adj, directed=True).to_dense(fill_value=0), expected.toarray())


@pytest.mark.parametrize("n", list(range(3, 10)))
def test_triangle_complete(n: int):
    adj_scipy = to_scipy(nx.complete_graph(n))
    expected = np.ones(adj_scipy.shape, dtype=adj_scipy.dtype)
    expected = (n - 2) * (expected - np.eye(expected.shape[0])) 
    adj = gb.io.from_scipy_sparse(adj_scipy)
    assert np.allclose(triangle(adj, directed=False).to_dense(fill_value=0), expected)