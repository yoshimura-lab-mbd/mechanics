from dataclasses import dataclass
from typing import Any, Optional, cast
import sympy as sp

from mechanics.symbol import ExplicitEquations, Variable, Expr, variable
from mechanics.conversion import Conversion, Replacement
from mechanics.group import group_from_mapping, group_key

@dataclass(frozen=True)
class FirstOrderResult:
    equations: ExplicitEquations
    variables: tuple[Variable, ...]
    conversion: Conversion

    _variables_ordered: dict[int, Any]

    def variables_of_order(self, order: int) -> Any:
        """ Get the variables corresponding to the n-th order derivatives. """
        return self._variables_ordered[order]



def to_first_order(F: ExplicitEquations, t: Optional[Variable] = None,
                   diff_var_prefixes: list[str] = ['v', 'a']) -> FirstOrderResult:
    """ Convert higher-order explicit equations to first-order explicit equations.
    
    Args:
        F: Explicit N-th order differential equations.
        t: Time variable. If None, guess from F.
        diff_var_prefixes: List of prefixes for derivative variables, e.g., ['v', 'a'] for velocity and acceleration.
    """

    new_vars: dict[sp.Derivative, Variable] = {}
    diffs: ExplicitEquations = {}
    variables_ordered: dict[int, dict[str, Variable]] = {}

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

            diff_var = variable(diff_name, *var.index_subs.keys(), *var.base_spaces, space=var.space)
            new_vars[sp.Derivative(var.general_form(), (t, n))] = diff_var
            key = group_key(var.name)
            variables_ordered.setdefault(n, {})
            if key in variables_ordered[n]:
                raise ValueError(f'Duplicate python_name key "{key}" for variable "{var.name}".')
            variables_ordered[n][key] = diff_var.general_form()
            diff_var_ = diff_var.subs(var.arg_subs) # type: ignore
            diff_vars[n] = diff_var_
            diffs[sp.Derivative(diff_vars[n-1], (t, 1))] = diff_var_

    subs = list(reversed(new_vars.items()))
    rep = Replacement(subs) # type: ignore

    equations = {cast(Variable, rep(dx)): rep(fX) for dx, fX in F.items()} | diffs
    variables = tuple(v for order in sorted(variables_ordered) for v in variables_ordered[order].values())

    variables_ordered_namedtuple = {}
    for order, mapping in variables_ordered.items():
        variables_ordered_namedtuple[order] = group_from_mapping(mapping, typename=f"VariablesOrder{order}")

    return FirstOrderResult(
        equations=equations,
        variables=variables,
        conversion=rep,
        _variables_ordered=variables_ordered_namedtuple,
    )
    

def is_first_derivative(expr: Expr) -> bool:
    return isinstance(expr, sp.Derivative) and len(expr.args) == 2 and expr.args[1][1] == 1 # type: ignore


def is_first_order(F: ExplicitEquations) -> bool:
    """Check if all equations in F are first-order differential equations."""
    return all(is_first_derivative(dx) for dx in F.keys())
