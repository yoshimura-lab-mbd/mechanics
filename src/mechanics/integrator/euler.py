from typing import Optional, cast

from mechanics.function import ExplicitEquations, Function, Expr, Index, BaseSpace
from mechanics.differential import is_first_derivative, to_first_order
from mechanics.discretization import discretization
from mechanics.conversion import Conversion

# \dot{X} = F(X)
def euler_explicit(F: ExplicitEquations, dt: Expr, i: Index) \
    -> tuple[tuple[Function, ...], ExplicitEquations, Conversion]:
    F, r = to_first_order(F)
    
    t: Optional[BaseSpace] = None
    for dx, fX in F.items():
        if not is_first_derivative(dx):
            raise NotImplementedError('Only first-order derivatives are supported in Euler integrator, got: ' + str(dx))
        t = dx.args[1][0]  # type: ignore
    if t is None:
        raise ValueError('No time variable found in equations.')

    d = discretization().space(t, i, dt)

    X = []
    step = {}
    for dx, fX in F.items():
        x_ = d(cast(Expr, dx.args[0]))
        X.append(x_)
        step[x_.subs(i, i + 1)] = x_ + dt * d(fX)

    return tuple(X), step, d * r
