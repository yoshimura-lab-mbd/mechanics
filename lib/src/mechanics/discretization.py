from collections import defaultdict
from typing import Self, cast, overload
import sympy as sp

from mechanics.conversion import Conversion
from mechanics.difference import FiniteDifference

from .symbol import BaseSpace, Index, Expr, Variable, variables, shift_index

class Discretization(Conversion):

    _indices: dict[BaseSpace, Index]
    _steps: dict[BaseSpace, Expr]
    _zero_index: dict[BaseSpace, Expr]
    _diffs: defaultdict[BaseSpace, dict[int, FiniteDifference]]

    def __init__(self):
        self._indices = {}
        self._steps = {}
        self._zero_index = {}
        self._diffs = defaultdict(dict)

    def space(self, space: BaseSpace, index: Index, step: Expr, 
              zero_index: Expr = sp.S.Zero) -> Self:
        self._indices[space] = index
        self._steps[space] = step
        self._zero_index[space] = zero_index
        return self

    def diff(self, space: BaseSpace, *diffs: FiniteDifference) -> Self:
        if space not in self._indices:
            raise ValueError(f'Space {space} not defined in discretization. '
                              'Define it using .space() method before adding differences.')
        for diff in diffs:
            self._diffs[space][diff.order] = diff

        return self

    def _replace_diff(self, *args) -> Expr:
        expr = cast(Variable, args[0])
        keeps = []
        for s, n in args[1:]:
            if s not in self._indices:
                keeps.append((s, n))
                continue

            index = self._indices[s]
            step = self._steps[s]
            diffs = self._diffs[s]
            if n not in diffs:
                raise ValueError(f'No finite difference of order {n} defined for space {s}. '
                                  'Define it using .diff() method before applying differentiation.')

            def var_shifted(shift: int) -> Expr:
                return shift_index(expr, index, shift)

            expr = diffs[n].apply(var_shifted, step)

            
        return sp.Derivative(expr, *keeps) if keeps else expr

    def _replace_variable(self, var: Variable) -> Variable:
        new_base_spaces = []
        new_indices = []
        new_subs = {}

        for s, value in var.base_space_subs.items():
            if s in self._indices:
                i = self._indices[s]
                step = self._steps[s]
                new_indices.append(i)
                if value == s:
                    pass
                else:
                    new_subs[i] = (value - self._zero_index[s]) / step

            else:
                new_base_spaces.append(s)
                new_subs[s] = value

        for i, value in var.index_subs.items():
            new_subs[i] = value

        new_var, = variables(var.name, 
                             *var.index_subs.keys(), *tuple(new_indices), 
                             space=var.space, base_spaces=tuple(new_base_spaces))

        return cast(Variable, new_var.subs(var.index_subs.items()).subs(new_subs))

    def _replace_subs(self, *args) -> Expr:
        expr = args[0]
        new_subs = []
        for var, value in zip(args[1], args[2]):
            if isinstance(var, BaseSpace) and var in self._indices:
                new_subs.append((self._indices[var], (value - self._zero_index[var]) / self._steps[var]))
            else:
                new_subs.append((var, value))
        return expr.subs(new_subs)

    def convert_expr(self, expr: Expr) -> Expr:
        expr = sp.sympify(expr)
        expr = cast(Expr, expr.replace(lambda e: isinstance(e, Variable), self._replace_variable))
        expr = cast(Expr, expr.replace(sp.Derivative, self._replace_diff))
        expr = cast(Expr, expr.replace(sp.Subs, self._replace_subs))
        return expr

def discretization() -> Discretization:
    return Discretization()