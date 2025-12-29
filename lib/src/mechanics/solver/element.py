from typing import Any
import sympy as sp

from mechanics.util import tuple_ish, to_tuple
from mechanics.symbol import Variable, Index, Expr, IndexRange
from .runner import ErrorReceiver
from .fortran import FortranPrinter


class SolverContext:
    error_receiver: ErrorReceiver

    variables: dict[Variable, tuple[IndexRange, ...]]
    functions: dict[str, tuple[IndexRange, ...]]
    constants: list[Variable]
    indices: set[Index]

    newton_sizes: list[Expr]

    check_options: dict[str, bool]

    def __init__(self, receiver: ErrorReceiver, check_options: dict[str, bool] = {}) -> None:
        self.error_receiver = receiver
        self.variables = {}
        self.constants = []
        self.indices = set()
        self.functions = {}
        self.newton_sizes = []
        self.check_options = check_options

    def register_indices(self, *index: Index):
        self.indices.update(index)

    def require_newton_size(self, size: Expr):
        self.newton_sizes.append(size)

    def ranges_of(self, var: Variable | str) -> tuple[IndexRange, ...]:
        if isinstance(var, str):
            var_ranges = self.functions.get(var, None)
            if var_ranges is None:
                raise ValueError(f'Function {var} not defined in solver.')
            indices = {range.index: range.index for range in var_ranges}
        else:
            var_ranges = self.variables.get(var.general_form(), None)
            if var_ranges is None:
                raise ValueError(f'Variable {var} not defined in solver.')
            indices = var.index_subs
        
        ranges = []
        for range in var_ranges:
            if indices[range.index] == range.index:
                ranges.append(range)
        return tuple(ranges)

    def shape_of(self, var: Variable | str, subs: Any = {}) -> tuple[Expr, ...]:
        shape = []
        for range in self.ranges_of(var):
            shape.append(sp.sympify(range.end - range.start + 1).subs(subs))
        return tuple(shape)

    def size_of(self, var: Variable) -> Expr:
        size = sp.S.One
        for n in self.shape_of(var):
            size *= n
        return size

    def shape_in_context_of(self, var: Variable, indices: list[tuple[Index, Expr, Expr]]) -> tuple[Expr, ...]:
        index_dict = {i: (start, end) for i, start, end in indices}
        shape = []
        for i, i_value in var.index_subs.items():
            if i == i_value:
                if i in index_dict:
                    start, end = index_dict[i]
                    shape.append(sp.sympify(end - start + 1))
                else:                   
                    raise ValueError(f'Index {i} not defined in context for variable {var}.')
            else:
                shape.append(sp.S.One)
        return tuple(shape)

    def bound_condition_of(self, var: Variable) -> Expr:
        ranges = self.ranges_of(var)
        condition = sp.S.true
        for range in ranges:
            condition = condition & ((range.start <= range.index) & (range.index <= range.end))
        return condition

class SolverElement(ErrorReceiver):

    _context: SolverContext

    def __init__(self, context: SolverContext, root: bool = False) -> None:
        if root:
            super().__init__()
        else:
            super().__init__(context.error_receiver)
        self._context = context

    def _generate(self, printer: FortranPrinter) -> str:
        return ''

