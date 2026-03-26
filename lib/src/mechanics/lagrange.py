import itertools
from typing import cast, Optional
import sympy as sp
from collections import defaultdict

from mechanics.conversion import Conversion, Replacement
from .symbol import Expr, ImplicitEquations, Variable, BaseSpace, variable, shift_index, Index, IndexRange, IndexRanges
from .util import tuple_ish, to_tuple, sympify


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
        p_n = variable(f'p_{{{q_n.name}}}', *q_n.base_spaces)
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



def euler_lagrange_equation(L: Expr, q: tuple_ish[Variable]) -> ImplicitEquations:
    """
    Euler-Lagrange equations for Lagrangian L with respect to generalized coordinates q.
    Parameters
    ----------
    L : Expr
        The Lagrangian expression.
    q : tuple_ish[Variable]
        The generalized coordinates. All q must have same one base space representing time.
    Returns
    -------
    ImplicitEquations
        The Euler-Lagrange equations.
    """

    t = None

    for q_n in to_tuple(q):
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

    equations: list[Expr] = []

    max_order = 1
    for d in L.atoms(sp.Derivative):
        for s, n in d.args[1:]: # type: ignore
            if s == t:
                max_order = max(max_order, n)

    for q_n in to_tuple(q):
        # for d in L.atoms(sp.Derivative):
        #     print(d.args)

        dLdq = sp.diff(L, q_n)
        eq: Expr = dLdq

        for order in range(1, max_order + 1):
            dLddnq = sp.diff(L, sp.diff(q_n, t, order))
            d_dLddnq_dnt = sp.diff(dLddnq, t, order)
            eq += (-1)**order * d_dLddnq_dnt

        equations.append(eq)

    return tuple(equations)



def euler_lagrange_field_equation(L: Expr, q: tuple_ish[Variable])\
    -> tuple[ImplicitEquations, dict[BaseSpace, ImplicitEquations]]:
    """
    Euler-Lagrange field equations for Lagrangian density L with respect to generalized fields q.
    Parameters
    ----------
    L : Expr
        The Lagrangian density expression.
    q : tuple_ish[Variable]
        The generalized fields. It can have multiple base spaces representing spacetime.
    Returns
    -------
    tuple[ImplicitEquations, dict[BaseSpace, ImplicitEquations]]
        A tuple containing:
        - The Euler-Lagrange field equations.
        - The boundary equations for each base space.
    """
    
    equations: list[Expr] = []
    boundaries: dict[BaseSpace, list[Expr]] = defaultdict(list)

    order_patterns: set[tuple[tuple[BaseSpace, int]]] = set()
    order_patterns.add(()) # type: ignore
    for d in L.atoms(sp.Derivative):
        order_patterns.add(tuple((s, n) for s, n in d.args[1:] if isinstance(s, BaseSpace))) # type: ignore

    for q_n in to_tuple(q):
        eq: Expr = sp.S.Zero

        for orders in order_patterns:
            dLdv = sp.diff(L, sp.diff(q_n, *orders) if orders else q_n)
            d_dLdv_ds = sp.diff(dLdv, *orders) if orders else dLdv
            order = sum(n for s, n in orders)
            eq += (-1)**order * d_dLdv_ds

            for s in q_n.base_spaces:
                bc: Expr = sp.S.Zero
                for orders_bc in order_patterns:
                    order_diff = dict(orders_bc)
                    order_less = False
                    for t, n in orders:
                        if t not in order_diff:
                            order_less = True
                            break
                        n_ = order_diff[t] - n
                        if n_ < 0:
                            order_less = True
                            break
                        elif n_ == 0:
                            del order_diff[t]
                        else:
                            order_diff[t] = n_
                    if order_less:
                        continue

                    dLda = sp.diff(L, sp.diff(q_n, (s, 1), *orders_bc))
                    d_dLda_ds = sp.diff(dLda, *orders_bc) if orders_bc else dLda
                    order = sum(n for s, n in orders_bc)
                    bc += (-1)**order * d_dLda_ds

                if bc != sp.S.Zero:
                    boundaries[s].append(bc)

        equations.append(eq)

    return tuple(equations), {s: tuple(bc) for s, bc in boundaries.items()}

    
def discrete_euler_lagrange_equation(L: Expr, q: tuple_ish[Variable]) -> ImplicitEquations:
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


def discrete_euler_lagrange_field_equation(
    L: Expr, q: tuple_ish[Variable], 
    *indices: tuple[Index, Expr | int, Expr | int],
    fixed: tuple_ish[tuple[Index, Expr | int]] = tuple(),
    )\
    -> tuple[tuple[Expr, IndexRanges], ...]:
    """
    Discrete Euler-Lagrange field equations for discrete Lagrangian density L with respect to generalized fields q.

    Parameters
    ----------
    L : Expr
        The discrete Lagrangian density.
    q : tuple_ish[Variable]
        The generalized fields.
    indices : tuple[Index, Expr | int, Expr | int]
        The index ranges for the discrete fields.
    fixed : tuple_ish[tuple[Index, Expr | int]]
        The fixed indices and their values.
    Returns
    -------
    tuple[tuple[Expr, ...], dict[BaseSpace, tuple[Expr, ...]]]
        A tuple containing:
        - The discrete Euler-Lagrange field equations.
        - The boundary equations for each base space.
    """

    q = to_tuple(q)
    ranges: dict[Index, IndexRange] = {}
    for index, start, end in indices:
        ranges[index] = IndexRange(index, sympify(start), sympify(end))
    fixed = to_tuple(fixed)

    appearances: defaultdict[Variable, set[Variable]] = defaultdict(set)
    for v in L.atoms(Variable):
        if v.general_form() in q:
            appearances[v.general_form()].add(v)

    equations: list[tuple[Expr, IndexRanges]] = []

    for q_n in q:
        if q_n.base_spaces:
            raise ValueError('Generalized fields q must be fully discrete (no base spaces).')
        for i in q_n.indices:
            if i not in ranges:
                raise ValueError(f'Index {i} of generalized field {q_n} not defined in index ranges {indices}.')

        eq: Expr = sp.S.Zero

        for q_i in appearances[q_n.general_form()]:
            dLdq = sp.diff(L, q_i)
            shifted = dLdq
            for i, i_value in q_i.index_subs.items():
                shift = int(i_value - i)
                shifted = shift_index(shifted, i, -shift)
            eq += shifted

        equations.append((eq, tuple(ranges.values())))

    return tuple(equations)
