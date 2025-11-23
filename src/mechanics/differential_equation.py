from typing import Optional, cast
import sympy as sp
from sympy.strategies import new

from mechanics.symbol import ExplicitEquations, Variable, Expr, variables
from mechanics.conversion import Conversion, Replacement


def to_first_order(F: ExplicitEquations, t: Optional[Variable] = None, 
                   diff_var_prefixes: list[str] = ['v', 'a']) -> tuple[ExplicitEquations, Conversion]:
    """ Convert higher-order explicit equations to first-order explicit equations.
    
    Args:
        F: Explicit N-th order differential equations.
        t: Time variable. If None, guess from F.
        diff_var_prefixes: List of prefixes for derivative variables, e.g., ['v', 'a'] for velocity and acceleration.
    """

    new_vars = {}
    diffs = {}

    for dx, fX in F.items():
        if not isinstance(dx, sp.Derivative):
            continue

        if len(dx.args) != 2:
            raise NotImplementedError('Only single variable derivatives are supported, got: ' + str(dx.args))

        if not isinstance(dx.args[0], Variable):
            raise NotImplementedError('Only derivatives of functions are supported, got: ' + str(var))

        var = cast(Variable, dx.args[0])
        t, order = dx.args[1] # type: ignore

        # if sp.Derivative(var, (t, 1)) in new_vars:
        #     continue

        diff_vars = {0: var}

        for n in range(1, order):
            if n >= len(diff_var_prefixes):
                diff_name = f'{var.name}_{n}'
            else:
                diff_name = f'{diff_var_prefixes[n - 1]}_{{{var.name}}}'

            diff_var, = variables(diff_name, *var.index_subs.keys(), *var.base_spaces, space=var.space)
            new_vars[sp.Derivative(var.general_form(), (t, n))] = diff_var
            diff_var_ = diff_var.subs(var.arg_subs) # type: ignore
            diff_vars[n] = diff_var_
            diffs[sp.Derivative(diff_vars[n-1], (t, 1))] = diff_var_

    subs = list(reversed(new_vars.items()))
    rep = Replacement(subs)

    return { cast(Variable, rep(dx)): rep(fX) for dx, fX in F.items() } | diffs, rep
    

def is_first_derivative(expr: Expr) -> bool:
    return isinstance(expr, sp.Derivative) and len(expr.args) == 2 and expr.args[1][1] == 1 # type: ignore