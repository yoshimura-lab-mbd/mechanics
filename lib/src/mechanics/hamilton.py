from operator import eq
from typing import cast

import sympy as sp
from .symbol import ExplicitEquations, Expr, Variable, BaseSpace, Index, variables
from .space import Space, R
from .util import tuple_ish, to_tuple

def hamiltons_equation(H: Expr, q: tuple_ish[Variable], p: tuple_ish[Variable]) -> ExplicitEquations:
    """
    Hamilton's equations for Hamiltonian H with respect to generalized coordinates q and momenta p.
    Parameters
    ----------
    H : Expr
        The Hamiltonian expression.
    q : tuple_ish[Variable]
        The generalized coordinates. 
    p : tuple_ish[Variable]
        The generalized momenta corresponding to q.
    Returns
    -------
    tuple[Expr, ...]
        The Hamilton's equations.
    """

    if len(to_tuple(q)) != len(to_tuple(p)):
        raise ValueError('The number of generalized coordinates q must match the number of generalized momenta p.')

    for q_n in to_tuple(q) + to_tuple(p):
        if len(q_n.base_spaces) != 1:
            raise ValueError('Each generalized coordinate q must have exactly one base space representing time.')
        s = q_n.base_spaces[0]
        if not isinstance(s, BaseSpace):
            raise ValueError('The base space of generalized coordinates q must be pure BaseSpace representing time.')
        if t is None:
            t = s
        elif t != s:
            raise ValueError('All generalized coordinates q must share the same base space representing time.')

    t = cast(BaseSpace, t)

    equations = {}

    for q_n, p_n in zip(to_tuple(q), to_tuple(p)):
        equations[sp.diff(q_n, t)] = sp.diff(H, p_n)
        equations[sp.diff(p_n, t)] = -sp.diff(H, q_n)

    return equations

def coordinates(name, *indices: Index, space: Space = R, **options) -> tuple[tuple[Variable, ...], tuple[Variable, ...]]:
    """Generate generalized coordinates and momenta variables.
    Parameters
    ----------
    name : str
        The base name(s) for the variables. If multiple variables are desired, separate names by spaces.
    indices : Index
        The indices for the variables.
    space : Space, optional
        The space for the variables, by default R.
    options : dict
        Additional options for the variables.
    Returns
    -------
    tuple[tuple[Variable, ...], tuple[Variable, ...]]
        The generalized coordinates and momenta variables.
    """

    q_ns = variables(name, *indices, space=space, **options)
    q = tuple(q_ns)

    p_names = ' '.join([f'p_{{{q_n.name}}}' for q_n in q])
    p_ns = variables(p_names, *indices, space=space, **options)
    p = tuple(p_ns)
    return q, p
