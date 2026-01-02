from typing import cast, Optional
import sympy as sp
from collections import defaultdict

from mechanics.conversion import Conversion, Replacement
from .symbol import Expr, Variable, BaseSpace, variables, shift_index, Index
from .util import tuple_ish, to_tuple


def is_regular_lagrangian(L: Expr, v: tuple_ish[Variable]) -> bool:
    """
    Check if the Lagrangian L is regular with respect to generalized velocities dq.

    Parameters
    ----------
    L : Expr
        The Lagrangian expression.
    v : tuple_ish[Variable]
        The generalized velocities.

    Returns
    -------
    bool
        True if the Lagrangian is regular, False otherwise.
    """

    p = []
    for v_n in to_tuple(v):
        p_n = sp.diff(L, v_n)
        p.append(p_n)

    J = sp.Matrix([[sp.diff(p_i, v_j) for v_j in to_tuple(v)] for p_i in p])
    return J.det() != 0


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

def legendre_transform(L: Expr, q: tuple_ish[Variable], v: tuple_ish[Variable]) \
    -> tuple[dict[Variable, Expr], Expr, Conversion]:
    """
    Legendre transform of Lagrangian L with respect to generalized velocities dq.

    Parameters
    ----------
    L : Expr
        The Lagrangian expression.
    q : tuple_ish[Variable]
        The generalized coordinates.
    v : tuple_ish[Variable]
        The generalized velocities.

    Returns
    -------
    tuple[dict[Variable, Expr], Expr, Conversion]
        A tuple containing:
        - The dictionary of conjugate momenta p, where each p_n = dL/dv_n. 
        - The Hamiltonian expression H = <p, v> - L.
        - A Conversion object representing the Legendre transform.
    """

    q = to_tuple(q)
    v = to_tuple(v)

    if len(q) != len(v):
        raise ValueError('Length of generalized coordinates q and velocities dq must be the same.')
    
    # Conjugate momenta
    p = []
    p_qv = []
    for q_n, v_n in zip(q, v):
        p_n, = variables(f'p_{{{q_n.name}}}', *q_n.base_spaces)
        p.append(p_n)
        p_qv.append(sp.diff(L, v_n))

    # Check regularity
    J = sp.Matrix([[sp.diff(p_i, v_j) for v_j in to_tuple(v)] for p_i in p_qv])
    if J.det() == 0:
        raise ValueError('Lagrangian is not regular with respect to the given generalized velocities.')

    # Calculate inverse transformation v(p)
    v_solved = sp.solve([p_n - p_n_qv for p_n, p_n_qv in zip(p, p_qv)], v)
    conversion = Replacement(v_solved)

    # Hamiltonian
    H = sum(p_n * v_solved[v_n] for p_n, v_n in zip(p, v)) - conversion(L)

    return dict(zip(p, p_qv)), H, conversion
    
    
def discrete_euler_lagrange_equation(L: Expr, q: tuple_ish[Variable]) -> tuple[Expr, ...]:
    """
    Discrete Euler-Lagrange equations for discrete Lagrangian L with respect to generalized coordinates q.

    Parameters
    ----------
    L : Expr
        The discrete Lagrangian. 
    q : tuple_ish[Variable]
        The generalized coordinates.
    Returns
    -------
    tuple[Expr, ...]
        The discrete Euler-Lagrange equations.
    """

    q = to_tuple(q)

    i: Optional[Index] = None

    for q_n in q:
        if q_n.base_spaces:
            raise ValueError('Generalized coordinates q must be fully discrete (no base spaces).')
        if len(q_n.indices) != 1:
            raise ValueError('Generalized coordinates q must have exactly one index representing time steps.')
        if i is None:
            if not isinstance(q_n.indices[0], Index):
                raise ValueError('The index of generalized coordinates q must be pure Index.')
            i = q_n.indices[0]
        elif i != q_n.indices[0]:
            raise ValueError('All generalized coordinates q must share the same time index.')
    
    i = cast(Index, i)

    appearances: defaultdict[Variable, set[Variable]] = defaultdict(set)

    for v in L.atoms(Variable):
        if v.general_form() in q:
            appearances[v.general_form()].add(v)

    equations: list[Expr] = []

    for q_n in q:
        eq = sp.S.Zero
        for q_i in appearances[q_n]:
            shift = int(q_i.indices[0] - i)
            dLdq = sp.diff(L, q_i)
            eq += shift_index(dLdq, i, -shift)
        equations.append(eq)

    return tuple(equations)