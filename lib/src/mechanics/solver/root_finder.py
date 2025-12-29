
from typing import Self, Optional
import sympy as sp
import textwrap
import inspect

from mechanics.symbol import ExplicitEquations, ImplicitEquations, IndexRange, Variable, Expr, Index
from mechanics.util import sympify, format_frameinfo
from .element import SolverElement, SolverContext
from .fortran import FortranPrinter


class RootFinder(SolverElement):
    _equations: list[tuple[Expr, tuple[IndexRange, ...]]]
    _unknowns: dict[Variable, tuple[IndexRange, ...]]
    _jacobian: dict[Variable, list[Expr]]
    _equations_size: Expr
    _unknowns_size: Expr

    def __init__(self, context: SolverContext) -> None:
        super().__init__(context)

        self._equations = []
        self._unknowns = {}


    def unknowns(self, *unknowns: Variable, 
                  index: Optional[tuple[Index, Expr | int, Expr | int]] = None,
                  indices: list[tuple[Index, Expr | int, Expr | int]] = []) -> Self:
        if index:
            indices = [index] + indices
        index_ranges = tuple(IndexRange(i, sympify(lower), sympify(upper)) 
                             for i, lower, upper in indices)
        self._unknowns.update({var: index_ranges for var in unknowns})
            
        return self

    def explicit(self, equations: ExplicitEquations,
                 index: Optional[tuple[Index, Expr | int, Expr | int]] = None,
                 indices: list[tuple[Index, Expr | int, Expr | int]] = []) -> Self:
        if index:
            indices = [index] + indices
        index_ranges = tuple(IndexRange(i, sympify(lower), sympify(upper)) 
                             for i, lower, upper in indices)
        self._equations.extend((f - eq, index_ranges) for f, eq in equations.items())
        return self
    
    def implicit(self, equations: ImplicitEquations,
                 index: Optional[tuple[Index, Expr | int, Expr | int]] = None,
                 indices: list[tuple[Index, Expr | int, Expr | int]] = []) -> Self:
        if index:
            indices = [index] + indices
        index_ranges = tuple(IndexRange(i, sympify(lower), sympify(upper)) 
                             for i, lower, upper in indices)
        self._equations.extend((eq, index_ranges) for eq in equations)
        return self

    def _generate(self, printer: FortranPrinter) -> str:
        raise NotImplementedError()

    def _exit(self) -> None:
        self._equations_size = sp.S.Zero
        self._unknowns_size = sp.S.Zero
        for eq, index_ranges in self._equations:
            size = sp.S.One
            for range in index_ranges:
                size *= (range.end - range.start + 1)
            self._equations_size += size
        for var, index_ranges in self._unknowns.items():
            size = sp.S.One
            for range in index_ranges:
                size *= (range.end - range.start + 1)
            self._unknowns_size += size

        self._context.require_newton_size(self._unknowns_size)

        self._jacobian = {}

        for var in self._unknowns.keys():
            self._jacobian[var] = [sp.diff(eq, var) for eq, _ in self._equations]


class NewtonRootFinder(RootFinder):
    _initial_guess: ExplicitEquations
    max_iterations: int
    tol: float

    _frame_info: inspect.FrameInfo

    def __init__(self, context: SolverContext) -> None:
        super().__init__(context)

        self._initial_guess = {}
        self.max_iterations = 100
        self.tol = 1e-10

        self._frame_info = inspect.stack()[2]

    def initial_guess(self, values: ExplicitEquations) -> Self:
        self._initial_guess.update(values)
        return self

    @staticmethod
    def _dgesv_error(frame_info):
        raise ValueError(
            f'Failed to solve linear system for variable in Newton-Raphson root finder.\n'
            'at ' + format_frameinfo(frame_info)
        )

    @staticmethod
    def _convergence_error(frame_info):
        raise ValueError(
            f'Newton-Raphson root finder did not converge within the maximum number of iterations.\n'
            'at ' + format_frameinfo(frame_info)
        )

    def _generate(self, printer: FortranPrinter) -> str:
        p = printer.doprint

        code = f'''
        ! Newton-Raphson Root Finder
        '''
        for var, index_ranges in self._unknowns.items():
            size = sp.S.One
            for range in index_ranges:
                size *= (range.end - range.start + 1)

            initial_guess = self._initial_guess.get(var, sp.S.Zero)

            if size == sp.S.One:
                code += f'''
        {p(var)} = {p(initial_guess)}'''

            else:
                raise NotImplementedError('Initial guess for array unknowns is not implemented yet.')

        code += f'''

        do newton_iter = 1, {p(self.max_iterations)}
        '''

        n = sp.S.Zero
        for eq, index_ranges in self._equations:
            code += f'''
            newton_eq({p(n+1)}) = {p(eq)}'''
            n += sp.S.One

        code += f'''

            newton_residual = dnrm2({p(n)}, newton_eq, 1)
            if (newton_residual < {p(self.tol)}) exit

            newton_jac = 0d0
        '''

        n = sp.S.Zero
        for var, derivative in self._jacobian.items():
            m = sp.S.Zero
            for d in derivative:
                if d != sp.S.Zero:
                    code += f'''
            newton_jac({p(m+1)}, {p(n+1)}) = {p(d)}'''

                m += sp.S.One
            n += sp.S.One

        dgesv_error_key = self.register_error(self._dgesv_error, self._frame_info)
        code += f'''

            call dgesv({p(n)}, 1, newton_jac, {p(n)}, newton_ipiv, newton_eq, {p(n)}, newton_info)
            if (newton_info /= 0) then; error = {dgesv_error_key}; goto 999; end if;
        '''

        n = sp.S.Zero
        for var, index_ranges in self._unknowns.items():
            code += f'''
            {p(var)} = {p(var)} - newton_eq({p(n+1)})'''
            n += sp.S.One

        convergence_error_key = self.register_error(self._convergence_error, self._frame_info)
        code += f'''
        end do

        if (newton_iter >= {p(self.max_iterations)}) then; error = {convergence_error_key}; goto 999; end if;
        '''

        return textwrap.dedent(code.replace('&\n', '&\n          '))