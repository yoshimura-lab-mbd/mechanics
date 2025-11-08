from typing import Self, Any
import sympy as sp

from .function import BaseSpace, Index, Expr, Function

class Discretization():

    _space: dict[BaseSpace, tuple[Index, Expr, dict[Expr, Expr]]]

    def __init__(self):
        self._space = {}

    def space(self, space: BaseSpace, index: Index, step: Expr, 
              mapping: dict[Expr, Expr] = {sp.S.Zero: sp.S.Zero}) -> Self:
        self._space[space] = (index, step, mapping)
        return self

    def replace_function(self, f: Function) -> Function:
        new_base_spaces = []
        new_base_space_values = []
        new_indices = []
        for s, value in f.base_space_subs().items():
            if s in self._space:
                i, step, mapping = self._space[s]
                if value == s:
                    new_indices.append(i)
                elif value in mapping:
                    new_indices.append(mapping[value])
                else:
                    raise ValueError(f'Cannot map base space value {value} to index. Specify manually.')

            else:
                new_base_spaces.append(s)
                new_base_space_values.append(value)

        return Function.make(f.name, *f.indices, *new_indices, *new_base_space_values, 
                             space=f.space, base_spaces=tuple(new_base_spaces))

    def __call__(self, target: Any) -> Any:

        if isinstance(target, Expr):
            return target.replace(lambda expr: isinstance(expr, Function), self.replace_function)
        elif isinstance(target, dict):
            return { self(t): self(v) for t, v in target.items() }
        elif isinstance(target, tuple):
            return tuple(self(t) for t in target)
        elif isinstance(target, list):
            return [self(t) for t in target]
        else:
            raise NotImplementedError('Discretization not implemented for type: ' + str(type(target)))


def discretization() -> Discretization:
    return Discretization()