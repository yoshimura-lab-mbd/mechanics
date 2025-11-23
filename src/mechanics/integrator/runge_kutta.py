from typing import cast

from mechanics.symbol import Variable, Expr, ExplicitEquations, Index, variables
from mechanics.differential_equation import to_first_order
from mechanics.discretization import discretization
from mechanics.conversion import Conversion


# input: \dot{X} = F(X)
# output: K_1 = dt * F(X)
#         K_2 = dt * F(X + 1/2 * K_1)
#         K_3 = dt * F(X + 1/2 * K_2)
#         X_{i+1} = X_i + 1/6 * (K_1 + 2 * K_2 + 2 * K_3 + dt * F(X + K_3))
def rk4_explicit(F: ExplicitEquations, dt: Expr, i: Index) \
    -> tuple[tuple[Variable, ...], tuple[Variable],  ExplicitEquations, Conversion]:
    F, r = to_first_order(F)
    
    t = list(F.keys())[0].args[1][0]  # type: ignore
    d = discretization().space(t, i, dt)

    X = []
    K1 = {}
    K2 = {}
    K3 = {}
    F_ = {}
    step = {}
    for dx, fX in F.items():
        x_ = cast(Variable, d(dx.args[0]))
        X.append(x_)
        f_ = d(fX)
        F_[x_] = f_
        k1, k2, k3 = variables(f'k_{{1, {x_.name}}} k_{{2, {x_.name}}} k_{{3, {x_.name}}}', indices=x_.indices)
        K1[x_] = k1
        K2[x_] = k2
        K3[x_] = k3

    for x_, k1 in K1.items():
        step[k1] = dt * F_[x_]

    K1_subs = {x_: x_ + k1 / 2 for x_, k1 in K1.items()}

    for x_, k2 in K2.items():
        step[k2] = dt * F_[x_].subs(K1_subs)

    K2_subs = {x_: x_ + k2 / 2 for x_, k2 in K2.items()}

    for x_, k3 in K3.items():
        step[k3] = dt * F_[x_].subs(K2_subs)

    K3_subs = {x_: x_ + k3 for x_, k3 in K3.items()}
    
    for x_ in X:
        step[x_.subs(i, i + 1)] = x_ + (K1[x_] + 2 * K2[x_] + 2 * K3[x_] + dt * F_[x_].subs(K3_subs)) / 6

    return tuple(X), tuple(K1.values()) + tuple(K2.values()) + tuple(K3.values()), step, d * r