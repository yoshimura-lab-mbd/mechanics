from typing import Optional, cast

from mechanics.function import ExplicitEquations, Function, Expr, Index, BaseSpace
from mechanics.differential import is_first_derivative, to_first_order
from mechanics.discretization import discretization
from mechanics.conversion import Conversion

# input: \dot{X} = F(X)
# output: X_{i+1} = X_i + dt * F(X_i)
def euler_explicit(F: ExplicitEquations, dt: Expr, i: Index) \
    -> tuple[tuple[Function, ...], ExplicitEquations, Conversion]:
    F, r = to_first_order(F)
    
    t = list(F.keys())[0].args[1][0]  # type: ignore
    d = discretization().space(t, i, dt)

    X = []
    step = {}
    for dx, fX in F.items():
        x_ = d(cast(Expr, dx.args[0]))
        X.append(x_)
        step[x_.subs(i, i + 1)] = x_ + dt * d(fX)

    return tuple(X), step, d * r

# input: \dot{X} = F(X)
# output: K = dt * F(X)
#         X_{i+1} = X_i + 1/2 * (K + dt * F(X_i + K))
def modified_euler_explicit(F: ExplicitEquations, dt: Expr, i: Index) \
    -> tuple[tuple[Function, ...], tuple[Function],  ExplicitEquations, Conversion]:
    F, r = to_first_order(F)
    
    t = list(F.keys())[0].args[1][0]  # type: ignore
    d = discretization().space(t, i, dt)

    X = []
    K = {}
    F_ = {}
    step = {}
    for dx, fX in F.items():
        x_ = d(dx.args[0]) # type: ignore
        X.append(x_)
        f_ = d(fX)
        F_[x_] = f_
        k = Function.make(f'k_{{{x_.name}}}', *x_.indices)
        K[x_] = k
        step[k] = dt * f_

    K_subs = {x_: x_ + k for x_, k in K.items()}
    
    for x_, k in K.items():
        step[x_.subs(i, i + 1)] = x_ + (k + dt * F_[x_].subs(K_subs)) / 2

    return tuple(X), tuple(K.values()), step, d * r