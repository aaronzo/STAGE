from typing import Protocol, Any, cast

import graphblas.binary
import graphblas.core.matrix
import graphblas.core.vector
import graphblas.select
import graphblas.unary

class _GraphblasModule(Protocol):
    def __new__(cls) -> "_GraphblasModule":
        raise NotImplementedError
    class Matrix(graphblas.core.matrix.Matrix, Protocol): ...
    class Vector(graphblas.core.vector.Vector, Protocol): ...
    unary = graphblas.unary
    binary = graphblas.binary
    select = graphblas.select