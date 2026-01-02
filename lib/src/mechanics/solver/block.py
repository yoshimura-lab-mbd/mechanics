from typing import Self
from contextlib import contextmanager
import sympy as sp
import textwrap
import inspect

from mechanics.util import format_frameinfo, sympify
from mechanics.symbol import Variable, Index, ExplicitEquations, Expr
from .element import SolverElement, SolverContext, template_env
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

        template = template_env.get_template("block/explicit_element.f90")

        checked_bounds = set()
        checked_dependencies = set()

        equations = []

        for l, r in self._equations.items():
            bound_checks = {}
            dependency_checks = {}

            for var in r.atoms(Variable):
                if var in self._context.constants: continue

                if self._context.check_options.get('bounds', True):
                    bound_condition = self._context.bound_condition_of(var)
                    if bound_condition == sp.S.true:
                        pass
                    elif bound_condition == sp.S.false:
                        raise IndexError(f'Variable {var} out of bounds in explicit equation: {l} = {r}')
                    elif bound_condition not in checked_bounds:
                        bound_error_key = self.register_error(self._bound_error, self._frame_info, var, l, r)
                        bound_checks[p(bound_condition)] = bound_error_key
                        checked_bounds.add(bound_condition)

                if self._context.check_options.get('dependencies', True):
                    if var not in checked_dependencies:
                        dependency_error_key = self.register_error(self._dependency_error, self._frame_info, var, l, r)
                        dependency_checks[p(var)] = dependency_error_key
                        checked_dependencies.add(var)

            if self._context.check_options.get('nan', True):
                nan_error_key = self.register_error(self._nan_error, self._frame_info, l, r)
            else:
                nan_error_key = None

            checked_dependencies.add(l)

            equations.append({
                'l': p(l),
                'r': p(r),
                'bound_checks': bound_checks,
                'dependency_checks': dependency_checks,
                'nan_check': nan_error_key
            })

        return template.render(equations=equations)



class CalculationElement(SolverElement):
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
