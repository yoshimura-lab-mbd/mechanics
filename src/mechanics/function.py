from typing import Optional, cast, Any, Union
import sympy as sp
import sympy.core.function as spf
import sympy.core.relational as spr

from mechanics.util import to_tuple, tuple_ish, single_or_tuple, split_latex
from mechanics.space import Space, Z, R

Expr = sp.Expr
Basic = sp.Basic

class BaseSpace(sp.Symbol):

    def __new__(cls, name: str):
        return super().__new__(cls, name, real=True)

    def __init__(self, name: str):
        super().__init__()

class Index(sp.Symbol):
    def __new__(cls, name: str):
        return super().__new__(cls, name, integer=True)

    def assign(self, value: Expr) -> Expr:
        return value

class Function(spf.AppliedUndef):
    name: str
    _space:       Space
    _index_num:   int
    _iterable:    bool = False

    # Initialization

    @classmethod
    def make(cls, name: str, *args: Expr, space: Space = R, index_num: int = -1, **options) -> 'Function':
        if space == Z:
            options['integer'] = True
        return cast(Function, 
                    spf.UndefinedFunction(name, bases=(Function,), **options)
                        (*args, space=space, index_num=index_num, **options)) #type:ignore

    def __new__(cls, *args: Expr, space: Space = R, index_num: int = -1, **options):
        if index_num < 0:
            index_num = 0
            indices = []
            base_spaces = []
            for a in args:
                if isinstance(a, Index):
                    index_num += 1
                    indices.append(a)
                elif isinstance(a, BaseSpace):
                    base_spaces.append(a)
                else:
                    raise ValueError(f'Invalid argument {a} for Function {cls.__name__}, must be Index or BaseSpace')
            args = tuple(indices + base_spaces)

        var = cast(Function, super().__new__(cls, *args))

        if space == Z:
            var.is_Integer = True

        var._space = space
        var._index_num = index_num

        return var

    
    @property
    def func(self): #type:ignore
        return lambda *args, **options: \
            self.make(self.name, *args, space=self._space, index_num=self._index_num, **options)
    
    # Indexing
    def __getitem__(self, *index_values: Any) -> single_or_tuple['Function']:
        new_indices = list(self.indices)
        for n, value in enumerate(index_values):
            if n >= self._index_num:
                raise IndexError(f'Too many indices for function {self.name}, has only {self._index_num} indices')
            if isinstance(value, slice):
                if value.start is None and value.stop is None and value.step is None:
                    pass
                else:
                    raise NotImplementedError('Only empty slice is supported for indexing')
            else:
                new_indices[n] = value

        return self.make(self.name, *new_indices, *self.base_spaces, index_num=self._index_num, space=self._space)
   

   # Assigning base spaces 
    def at(self, *values: Any) -> single_or_tuple['Function']:
        new_base_spaces = list(self.base_spaces)
        for n, value in enumerate(values):
            if n >= len(new_base_spaces):
                raise IndexError(f'Too many base spaces for function {self.name}, has only {len(new_base_spaces)} base spaces')
            if isinstance(value, slice):
                if value.start is None and value.stop is None and value.step is None:
                    pass
                else:
                    raise NotImplementedError('Only empty slice is supported for base space assignment')
            else:
                new_base_spaces[n] = value
        return self.make(self.name, *self.indices, *new_base_spaces, index_num=self._index_num, space=self._space)

    # Printing

    def _sympystr(self, printer) -> str:
        return f'{self.name}{self.args}'

    def _latex(self, printer, exp=None, notation=None) -> str:
        latex = f'{self.name}'
        if notation:
            latex = notation(latex)
        if self.indices: 
            latex += f'_{{{",".join([sp.latex(i) for i in self.indices])}}}'
        if self.base_spaces and any(not isinstance(arg, BaseSpace) for arg in self.base_spaces):
            latex += f'\\left({",".join([sp.latex(e) for e in self.base_spaces])}\\right)'
        if exp: 
            latex = f'{{{latex}}}^{exp}'
        return latex
    
    # def __str__(self) -> str:
    #     return python_name(sp.latex(self))

    # Properties
      
    @property
    def base_spaces(self) -> tuple[Expr, ...]:
        return cast(tuple[Expr, ...], self.args[self._index_num:])
    
    @property
    def indices(self) -> tuple[Expr, ...]:
        return cast(tuple[Expr, ...], self.args[:self._index_num])
    
    @property
    def space(self) -> Space:
        return self._space
    
def base_spaces(name: str) -> tuple[BaseSpace, ...]:
    names = split_latex(name)
    return tuple(BaseSpace(n) for n in names)

def variables(name: str,
              *base_space_or_index: BaseSpace | Index,
              space: Space = R,
              **options) -> tuple[Function, ...]:
    names = split_latex(name)
    return tuple(Function.make(name, *base_space_or_index, space=space, **options)
                 for name in names)

def constants(name: str,
              *index: Index,
              space: Space = R,
              **options) -> tuple[Function, ...]:
    names = split_latex(name)
    return tuple(Function.make(name, *index, space=space, **options)
                 for name in names)




