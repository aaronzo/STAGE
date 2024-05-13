import graphblas as gb
from typing import TYPE_CHECKING, Optional
if TYPE_CHECKING:
    from gnn._typing import _GraphblasModule as gb
    from graphblas.core.matrix import MatrixExpression


def eval(
    expr: "MatrixExpression",
    out: Optional[gb.Matrix] = None,
    **kw
) -> gb.Matrix:
    if out is None:
        return expr.new(**kw)
    out(**kw) << expr
    return out