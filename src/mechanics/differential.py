from typing import Optional, cast
import sympy as sp
from sympy.strategies import new

from mechanics.function import ExplicitEquations, Function, Expr


def to_first_order(F: ExplicitEquations, t: Optional[Function] = None, diff_var_prefixes: list[str] = ['v', 'a']) -> ExplicitEquations:
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

        var = dx.args[0]
        if not isinstance(var, Function):
            raise NotImplementedError('Only derivatives of functions are supported, got: ' + str(var))
        t, order = dx.args[1] # type: ignore

        diff_vars = {0: var}

        for n in range(1, order):
            if n >= len(diff_var_prefixes):
                diff_name = f'{var.name}_{n}'
            else:
                diff_name = f'{diff_var_prefixes[n - 1]}_{var.name}'

            diff_var = Function.make(diff_name, *var.indices, *var.base_space_values, 
                                     base_spaces=var.base_spaces, space=var.space)
            new_vars[sp.Derivative(var, (t, n))] = diff_var
            diff_vars[n] = diff_var

            diffs[sp.Derivative(diff_vars[n-1], (t, 1))] = diff_var

    subs = list(reversed(new_vars.items()))

    return { cast(Function, dx.subs(subs)): fX.subs(subs) 
             for dx, fX in F.items() } | diffs
    

def is_first_derivative(expr: Expr) -> bool:
    return isinstance(expr, sp.Derivative) and len(expr.args) == 2 and expr.args[1][1] == 1 # type: ignore