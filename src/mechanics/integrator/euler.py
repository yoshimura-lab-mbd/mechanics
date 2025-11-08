import re
from typing import Optional

from mechanics.function import ExplicitEquations, Function, Expr, Index, BaseSpace
from mechanics.differential import is_first_derivative
from mechanics.discretization import discretization

# \dot{X} = F(X)
def euler_explicit(F: ExplicitEquations, dt: Expr, i: Index) -> ExplicitEquations:
    
    t: Optional[BaseSpace] = None
    for dx, fX in F.items():
        if not is_first_derivative(dx):
            raise NotImplementedError('Only first-order derivatives are supported in Euler integrator, got: ' + str(dx))
        t = dx.args[1][0]  # type: ignore
    if t is None:
        raise ValueError('No time variable found in equations.')

    d = discretization().space(t, i, dt)

    result = {}
    for dx, fX in F.items():
        x_ = d(dx.args[0])
        result[x_.subs(i, i + 1)] = x_ + dt * d(fX)

    return result
