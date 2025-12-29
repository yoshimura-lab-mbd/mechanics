from typing import Self
from contextlib import contextmanager
import sympy as sp
import textwrap
import inspect

from mechanics.util import format_frameinfo, sympify
from mechanics.symbol import Variable, Index, ExplicitEquations, Expr
from .element import SolverElement, SolverContext
from .fortran import FortranPrinter
from .root_finder import NewtonRootFinder



class DependencyError(ValueError):
    pass

class SolverBlock(SolverElement):
    _elements: list[SolverElement]

    def __init__(self, context: SolverContext, root: bool = False) -> None:
        super().__init__(context, root=root)
        self._elements = []

    def explicit(self, equations: ExplicitEquations) -> Self:
        element = ExplicitElement(self._context, equations)
        self._elements.append(element)
        return self

    def calculate(self, functions: dict[str, Expr], *indices: Index) -> Self:
        element = CalculationElement(self._context, functions, *indices)
        self._elements.append(element)
        return self

    @contextmanager
    def newton(self):
        block = NewtonRootFinder(self._context)
        yield block
        block._exit()
        self._elements.append(block)

    @contextmanager
    def steps(self, index: Index, start: Expr | int, end: Expr | int):
        block = StepsBlock(self._context, index, sympify(start), sympify(end))
        yield block
        block._exit()
        self._elements.append(block)

    def _exit(self) -> None:
        pass 

    def _generate(self, printer: FortranPrinter) -> str:
        code = ''
        for element in self._elements:
            code += element._generate(printer)
            code += '\n'
        return code


class ExplicitElement(SolverElement):
    _equations: ExplicitEquations

    _frame_info: inspect.FrameInfo

    def __init__(self, context: SolverContext, equations: ExplicitEquations) -> None:
        super().__init__(context)
        self._equations = equations

        self._frame_info = inspect.stack()[2]

        for l, r in equations.items():
            if not isinstance(l, Variable):
                raise TypeError(f'Left-hand side of explicit equation must be a Variable, got {type(l)}: {l} = {r}.')

            for var in r.atoms(Variable):
                if var.general_form() not in self._context.constants and \
                   var.general_form() not in self._context.variables:
                    raise ValueError(f'Variable {var} used before definition in explicit equation: {l} = {r}.')

    @staticmethod
    def _dependency_error(frame_info, var, l, r):
        raise DependencyError(
            f'Variable {var} used before calculation in explicit equation: {l} = {r}\n'
            'at ' + format_frameinfo(frame_info)
        )

    @staticmethod
    def _bound_error(frame_info, var, l, r):
        raise IndexError(
            f'Variable {var} out of bounds in explicit equation: {l} = {r}\n'
            'at ' + format_frameinfo(frame_info)
        )

    @staticmethod
    def _nan_error(frame_info, l, r):
        # raise ValueError(
        #     f'Calculation result is NaN in explicit equation: {l} = {r}\n'
        #     'at ' + format_frameinfo(frame_info)
        # )
        print(f'Warning: Calculation result is NaN in explicit equation: {l} = {r}.')

    def _generate(self, printer: FortranPrinter) -> str:
        p = printer.doprint

        code = '''
        ! Solve explicit equations
        '''

        checked_dependencies = set()

        for l, r in self._equations.items():
            checks = []
            for var in r.atoms(Variable):
                if var in self._context.constants: continue
                if var in checked_dependencies: continue

                checks.append(var)
                checked_dependencies.add(var)

                bound_condition = self._context.bound_condition_of(var)
                if bound_condition == sp.S.true:
                    pass
                elif bound_condition == sp.S.false:
                    raise IndexError(f'Variable {var} out of bounds in explicit equation: {l} = {r}')
                else:
                    bound_error_key = self.register_error(self._bound_error, self._frame_info, var, l, r)

                if self._context.check_options.get('bounds', True):
                    code += f'''
        if ({p(sp.Not(bound_condition))}) then; error = {bound_error_key}; goto 999; end if'''
                
                if self._context.check_options.get('dependencies', True):
                    dependency_error_key = self.register_error(self._dependency_error, self._frame_info, var, l, r)
                    code += f'''
        if (ieee_is_nan({p(var)})) then; error = {dependency_error_key}; goto 999; end if
        '''


            code += f'''
        {p(l)} = {p(r)}'''

            if self._context.check_options.get('nan', True):
                nan_error_key = self.register_error(self._nan_error, self._frame_info, l, r)
                code += f'''
        if (ieee_is_nan({p(l)})) then; error = {nan_error_key}; goto 999; end if
        '''

        return textwrap.dedent(code)


class CalculationElement(SolverBlock):
    _functions: dict[str, Expr]
    _indices: tuple[Index, ...]

    def __init__(self, context: SolverContext, functions: dict[str, Expr], *indices: Index) -> None:
        super().__init__(context)
        self._functions = functions
        self._indices = indices

    def _generate(self, printer: FortranPrinter) -> str:
        p = printer.doprint

        indices = ",".join([str(p(i)) for i in self._indices])

        code = '! Calculate functions\n'
        for name, f in self._functions.items():
            name = printer.fortran_name(name)
            if indices:
                code += f'{name}({indices}) = {p(f)}\n'
            else:
                code += f'{name} = {p(f)}\n'

        return code


class StepsBlock(SolverBlock):
    index: Index
    start: Expr | int
    end: Expr | int

    def __init__(self, context: SolverContext, index: Index, start: Expr, end: Expr) -> None:
        super().__init__(context)
        self.index = index
        self.start = start
        self.end = end

        context.register_indices(index)

    def _generate(self, printer: FortranPrinter) -> str:
        p = printer.doprint

        code = f'''
            ! Iterate over {p(self.index)}
            do {p(self.index)} = {p(self.start)}, {p(self.end)}'''

        for element in self._elements:
            code += '\n'
            code += textwrap.indent(element._generate(printer), '    ' * 4)

        code += f'''
            end do
        '''

        return textwrap.dedent(code)
