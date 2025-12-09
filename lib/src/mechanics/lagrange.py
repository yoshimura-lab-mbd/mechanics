import sympy as sp
from .symbol import Expr, Variable, BaseSpace
from .util import tuple_ish, to_tuple

def euler_lagrange_equation(L: Expr, q: tuple_ish[Variable]) -> tuple[Expr, ...]:
    base_spaces = set()
    for q_n in to_tuple(q):
        for t in q_n.base_spaces:
            if isinstance(t, BaseSpace):
                base_spaces.add(t)
    if not base_spaces:
        raise ValueError('No base spaces found in generalized coordinates')
    base_spaces = tuple(base_spaces)

    equations: list[Expr] = []

    for q_n in q:
        dLdq = sp.diff(L, q_n)
        eq: Expr = dLdq

        for t in base_spaces:
            dLddq = sp.diff(L, sp.diff(q_n, t))
            d_dLddq_dt = sp.diff(dLddq, t)
            eq -= d_dLddq_dt #type:ignore

        if eq != 0:
            equations.append(eq)

    return tuple(equations)