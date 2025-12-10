from typing import Optional, cast, Any, Union
import sympy as sp
import sympy.core.function as spf
import sympy.core.relational as spr
from sympy.matrices.repmatrix import index_
from sympy.strategies import new
from sympy.utilities.iterables import NotIterable


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

class IndexRange:
    index: Index
    start: Expr
    end:   Expr

    def __init__(self, index: Index, start: Expr, end: Expr) -> None:
        self.index = index
        self.start = start
        self.end = end

    def __str__(self) -> str:
        return f'{self.index} in [{self.start}, {self.end}]'

class Variable(spf.AppliedUndef, NotIterable):
    name: str
    args: tuple[Expr, ...]
    _original_indices: tuple[Index, ...]
    _base_spaces: tuple[BaseSpace, ...]
    _space:       Space

    # Initialization

    @classmethod
    def make(cls, name: str, *args: Expr, 
             original_indices: tuple[Expr, ...] = (),
             base_spaces: tuple[BaseSpace, ...] = (), 
             space: Space = R, 
             **options) -> 'Variable':
        if space == Z:
            options['integer'] = True

        if not original_indices:
            for arg in args:
                if isinstance(arg, Index):
                    original_indices += (arg,)
        if not base_spaces:
            for arg in args:
                if isinstance(arg, BaseSpace):
                    base_spaces += (arg,)

        return cast(Variable, 
                    spf.UndefinedFunction(name, bases=(Variable,), **options)
                        (*args, 
                        original_indices=original_indices,
                        base_spaces=base_spaces, 
                        space=space, 
                        **options)) #type:ignore

    def __new__(cls, *args: Expr, 
                original_indices: tuple[Index, ...] = (),
                base_spaces: tuple[BaseSpace, ...] = (),
                space: Space = R, 
                **options):
        var = cast(Variable, super().__new__(cls, *args))

        if space == Z:
            var.is_Integer = True

        var._original_indices = original_indices
        var._base_spaces = base_spaces
        var._space = space

        return var

    
    @property
    def func(self): #type:ignore
        return lambda *args, **options: \
            self.__class__(*args, 
                            original_indices=self._original_indices, 
                            base_spaces=self._base_spaces, 
                            space=self._space, **options)
    
    # Indexing
    def __getitem__(self, index_values: Any) -> 'Variable':
        if not isinstance(index_values, tuple):
            index_values = (index_values,)
        new_indices = []
        for n, value in enumerate(index_values):
            if n >= len(self.indices):
                raise IndexError(f'Too many indices for function {self.name}, has only {len(self.indices)} indices')
            if isinstance(value, slice):
                if value.start is None and value.stop is None and value.step is None:
                    new_indices.append(self.indices[n])
                else:
                    raise NotImplementedError('Only empty slice is supported for indexing')
            else:
                new_indices.append(value)

        if len(new_indices) < len(self.indices):
            new_indices += list(self.indices)[len(new_indices):]

        return self.make(self.name, *new_indices, *self.base_space_values, 
                         original_indices=self._original_indices,
                         space=self._space, base_spaces=self._base_spaces)
   
    # def __iter__(self):
    #     raise TypeError(f"{self.__class__.__name__} is not iterable")

   # Assigning base spaces 
    def at(self, *values: Any) -> single_or_tuple['Variable']:
        new_base_space_values = list(self.base_space_values)
        for n, value in enumerate(values):
            if n >= len(new_base_space_values):
                raise IndexError(f'Too many base spaces for function {self.name}, has only {len(self.base_spaces)} base spaces')
            if isinstance(value, slice):
                if value.start is None and value.stop is None and value.step is None:
                    pass
                else:
                    raise NotImplementedError('Only empty slice is supported for base space assignment')
            else:
                new_base_space_values[n] = value
        return self.make(self.name, *self.indices, *new_base_space_values, 
                        original_indices=self._original_indices, 
                        base_spaces=self._base_spaces,
                        space=self._space)

    def general_form(self) -> 'Variable':
        return self.make(self.name, *self._original_indices, *self.base_spaces, 
                         space=self._space)

    # Printing

    def _sympystr(self, printer) -> str:
        if self.args:
            return f'{self.name}{self.args}'
        return f'{self.name}'

    def _latex(self, printer, exp=None, notation=None) -> str:
        latex = f'{{{self.name}}}'
        if notation:
            latex = notation(latex)
        if self.indices: 
            latex += f'_{{{",".join([sp.latex(i) for i in self.indices])}}}'
        if exp: 
            latex = f'{{{latex}}}^{exp}'
        if self.base_spaces != self.base_space_values:
            subs = [f'{sp.latex(s)}={sp.latex(e)}' for s, e in self.base_space_subs.items() if s != e]
            latex += rf'\left|_{{{",".join(subs)}}}\right.'
        return latex
    
    def __str__(self) -> str:
        s = self.name
        if self.args:
            s += f'({", ".join([str(a) for a in self.args])})'
        return s

    # Properties

    @property
    def base_spaces(self) -> tuple[BaseSpace, ...]:
        return self._base_spaces
      
    @property
    def base_space_values(self) -> tuple[Expr, ...]:
        return cast(tuple[Expr, ...], self.args[len(self._original_indices):])

    @property
    def base_space_subs(self) -> dict[BaseSpace, Expr]:
        return {s: value for s, value in zip(self.base_spaces, self.base_space_values)}
    
    @property
    def indices(self) -> tuple[Expr, ...]:
        return cast(tuple[Expr, ...], self.args[:len(self._original_indices)])

    @property
    def index_subs(self) -> dict[Index, Expr]:
        return {i: value for i, value in zip(self._original_indices, self.indices)}

    @property
    def arg_subs(self) -> dict[Index | BaseSpace, Expr]:
        return {**self.index_subs, **self.base_space_subs}
    
    @property
    def space(self) -> Space:
        return self._space

    @property
    def is_constant(self) -> bool:
        return not self.base_spaces

    @property
    def _diff_wrt(self) -> bool:
        return not self.is_constant



ExplicitEquations = dict[Variable, Expr]
ImplicitEquations = tuple[Expr, ...]

    
def base_spaces(name: str) -> tuple[BaseSpace, ...]:
    names = split_latex(name)
    return tuple(BaseSpace(n) for n in names)

def indices(name: str) -> tuple[Index, ...]:
    names = split_latex(name)
    return tuple(Index(n) for n in names)

def variables(name: str,
              *args: Index | BaseSpace, 
              space: Space = R,
              **options) -> tuple[Variable, ...]:
    names = split_latex(name)
    return tuple(Variable.make(name, *args, space=space, **options)
                 for name in names)

def constants(name: str,
              *indices: Index,
              space: Space = R,
              **options) -> tuple[Variable, ...]:
    names = split_latex(name)
    return tuple(Variable.make(name, *indices, space=space, **options)
                 for name in names)


def to_implicit(F: ExplicitEquations) -> ImplicitEquations:
    return tuple(a - b for a, b in F.items())


def shift_index(expr: Expr, index: Index, shift: int) -> Expr:
    def replace_variable(var: Variable) -> Variable:
        index_subs = var.index_subs
        for i, value in index_subs.items():
            if i == index:
                index_subs[i] = value + shift
        return var[*index_subs.values()]

    return cast(Expr, expr.replace(lambda e: isinstance(e, Variable), replace_variable))
