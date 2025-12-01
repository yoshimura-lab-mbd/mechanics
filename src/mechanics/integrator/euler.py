from sys import displayhook
from typing import Optional, cast

from mechanics.symbol import ExplicitEquations, ImplicitEquations, Variable, Expr, Index, variables
from mechanics.differential_equation import to_first_order
from mechanics.discretization import discretization
from mechanics.conversion import Conversion

# input: \dot{X} = F(X)
# output: X_{i+1} = X_i + dt * F(X_i)
def euler_explicit(F: ExplicitEquations, dt: Expr, i: Index) \
    -> tuple[tuple[Variable, ...], ExplicitEquations, Conversion]:
    F, r = to_first_order(F)

    t = list(F.keys())[0].args[1][0]  # type: ignore
    d = discretization().space(t, i, dt)

    X = set()
    step = {}
    for dx, fX in F.items():
        x_ = cast(Variable, d(cast(Expr, dx.args[0])))
        X.add(x_.general_form())
        step[x_.subs(i, i + 1)] = x_ + dt * d(fX)

    return tuple(X), step, d * r

# input: \dot{X} = F(X)
# output: K = dt * F(X)
#         X_{i+1} = X_i + 1/2 * (K + dt * F(X_i + K))
def modified_euler_explicit(F: ExplicitEquations, dt: Expr, i: Index) \
    -> tuple[tuple[Variable, ...], tuple[Variable],  ExplicitEquations, Conversion]:
    F, r = to_first_order(F)
    
    t = list(F.keys())[0].args[1][0]  # type: ignore
    d = discretization().space(t, i, dt)

    X = []
    K = {}
    F_ = {}
    step = {}
    for dx, fX in F.items():
        x_ = cast(Variable, d(dx.args[0]))
        X.append(x_)
        f_ = d(fX)
        F_[x_] = f_
        k, = variables(f'k_{{{x_.name}}}', *x_.index_subs.keys())
        K[x_] = k

    for x_, k in K.items():
        step[k] = dt * F_[x_]

    K_subs = {x_: x_ + k for x_, k in K.items()}
    
    for x_ in X:
        step[x_.subs(i, i + 1)] = x_ + (K[x_] + dt * F_[x_].subs(K_subs)) / 2

    return tuple(X), tuple(K.values()), step, d * r

heun_explicit = modified_euler_explicit


# input: \dot{X} = F(X)
# output: X_{i+1} = dt * F(X_{i+1})
def backward_euler_explicit(F: ExplicitEquations, dt: Expr, i: Index) \
    -> tuple[tuple[Variable, ...], tuple[Variable, ...], ImplicitEquations, Conversion]:

    F, r = to_first_order(F)

    t = list(F.keys())[0].args[1][0]  # type: ignore
    d = discretization().space(t, i, dt)

    X = set()
    step = []
    unknowns = []
    for dx, fX in F.items():
        x_ = cast(Variable, d(cast(Expr, dx.args[0])))
        X.add(x_.general_form())
        step.append(x_.subs(i, i + 1) - x_ + dt * d(fX).subs(i, i + 1))
        unknowns.append(x_.subs(i, i + 1))

    return tuple(X), tuple(unknowns), tuple(step), d * r