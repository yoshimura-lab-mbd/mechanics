from typing import Self, Optional, cast, Any
from contextlib import contextmanager
import textwrap
import inspect
import sympy as sp

from mechanics.symbol import ExplicitEquations, ImplicitEquations, Variable, Expr, Index, IndexRange
from mechanics.util import format_frameinfo, sympify, to_tuple, tuple_ish, format_frameinfo
import mechanics.space as space
from sympy.matrices.repmatrix import index_
from .fortran import FortranPrinter
from .runner import SolverRunner
from .element import SolverElement, SolverContext
from .block import SolverBlock


class SolverBuilder(SolverBlock):
    _inputs: list[Variable]

    def __init__(self, check_options: dict[str, bool] = {}) -> None:
        super().__init__(SolverContext(self, check_options=check_options), root=True)

        self._inputs = []

        SolverBlock._key_counter = 0

    def constants(self, *constants: Variable,
                  index: Optional[tuple[Index, Expr | int, Expr | int]] = None,
                  indices: list[tuple[Index, Expr | int, Expr | int]] = []) -> Self:
        self.variables(*constants, index=index, indices=indices)
        self._context.constants.extend(constants)
        return self

    def variables(self, *variables: Variable, 
                  index: Optional[tuple[Index, Expr | int, Expr | int]] = None,
                  indices: list[tuple[Index, Expr | int, Expr | int]] = []) -> Self:
        if index:
            indices = [index] + indices

        index_ranges = tuple(IndexRange(i, sympify(lower), sympify(upper)) 
                             for i, lower, upper in indices)


        for range in index_ranges:
            if not isinstance(range.index, Index):
                raise TypeError(f"Expected Index, got {range.index} which has type {type(range.index)})")

        for var in variables:
            unknowns = self._unknowns_of(var.args)
            if unknowns:
                raise ValueError(f'{var} contains unknown variables or constants: {unknowns}. '
                                  'Add them using .variables() or .constants() first.')
            for range in index_ranges:
                unknowns = self._unknowns_of([range.start, range.end])
                if unknowns:
                    raise ValueError(f'Index specification {range} contains unknown constants: {unknowns}. '
                                      'Add them using .constants() first.')
            self._context.variables[var] = index_ranges

        return self

    def functions(self, *names: str, 
                  index: Optional[tuple[Index, Expr | int, Expr | int]] = None,
                  indices: list[tuple[Index, Expr | int, Expr | int]] = []) -> Self:
        if index:
            indices = [index] + indices

        index_ranges = tuple(IndexRange(i, sympify(lower), sympify(upper)) 
                             for i, lower, upper in indices)
        
        for range in index_ranges:
            if not isinstance(range.index, Index):
                raise TypeError(f"Expected Index, got {range.index} which has type {type(range.index)})")
        
        for name in names:
            self._context.functions[name] = index_ranges

        return self

    def inputs(self, *inputs: Variable) -> Self:
        self._inputs.extend(inputs)
        return self


    def generate(self) -> SolverRunner:
        printer = FortranPrinter({'source_format': 'free', 'strict': False, 'standard': 95, 'precision': 15})
        code = self._generate(printer)
        return SolverRunner(
            code, self,
            constants = {c: self._context.ranges_of(c) for c in self._context.constants},
            variables = {v: self._context.ranges_of(v) for v in self._context.variables},
            inputs = {v: self._context.ranges_of(v) for v in self._inputs},
            functions = {name: self._context.ranges_of(name) for name in self._context.functions}
        )

    def _unknowns_of(self, exprs: tuple_ish[Expr]) -> set[Variable]:
        unknowns = set()
        for expr in to_tuple(exprs):
            vars = sp.sympify(expr).atoms(Variable)
            for v in vars:
                if v not in self._context.constants and v not in self._inputs:
                    unknowns.add(v)
        return unknowns

    def _generate(self, printer: FortranPrinter) -> str:
        p = printer.doprint

        code = ''
        code += f'''
        subroutine run_solver(log_path, condition, error)
            use, intrinsic :: ieee_arithmetic
            implicit none
            double precision dnrm2
            external dnrm2

            character(len=*), intent(in) :: log_path
            real(8), dimension(:), intent(in) :: condition
            integer, intent(out) :: error

            integer :: input_n

            integer :: log_unit = 20
            integer :: ios

            ! Define constants and variables'''

        for var in self._context.variables:
            type_ = 'integer' if var.space == space.Z else 'real(8)'
            if var.indices:
                code += f'''
            {type_}, allocatable :: {printer.print_as_array_arg(var)}'''
            else:
                code += f'''
            {type_} :: {p(var)}'''

        code += '''

            ! Define functions'''

        for name, indices in self._context.functions.items():
            if indices:
                code += f'''
            real(8), allocatable :: {printer.fortran_name(name)}({", ".join([":" for _ in indices])})'''
            else:
                code += f'''
            real(8) :: {printer.fortran_name(name)}'''

        code += '''

            ! Define indices'''
        for index in self._context.indices:
            code += f'''
            integer :: {p(index)}'''

        code += '''

            ! Define Newton solver variables
            integer :: newton_size
            integer :: newton_iter
            real(8) :: newton_residual
            real(8), allocatable :: newton_eq(:)
            real(8), allocatable :: newton_jac(:,:)
            integer, allocatable :: newton_ipiv(:)
            integer :: newton_info

            ! Initialize variables
            input_n = 1
            '''
        
        initialized = set()

        for v in self._context.constants + self._inputs:
            var = v.general_form()
            ranges = self._context.variables.get(var, ())
            if var in self._context.variables and ranges and var not in initialized:
                initialized.add(var)
                name = printer.fortran_name(var.name)
                ranges_str = ",".join(f"{p(r.start)}:{p(r.end)}" for r in ranges)
                code += f'''
            allocate({name}({ranges_str}))
            {name} = ieee_value({name}, ieee_quiet_nan)'''

            name = printer.fortran_name(v.name)
            size = self._context.size_of(v)
            shape = self._context.shape_of(v)
            assign_to = printer.print_as_array_arg(v) if var != v else name

            if v.space == space.Z:
                if len(shape) == 0:
                    code += f'''
            {assign_to} = int(condition(input_n))
            input_n = input_n + 1
            '''

                elif len(shape) == 1: 
                    code += f'''
            {assign_to} = int(condition(input_n : input_n + {p(size)} - 1))
            input_n = input_n + {p(size)}
            '''

                else:
                    code += f'''
            {assign_to} = reshape(int(condition(input_n : input_n + {p(size)} - 1)), {p(shape)})
            input_n = input_n + {p(size)}
            '''


            else:
                if len(shape) == 0:
                    code += f'''
            {assign_to} = condition(input_n)
            input_n = input_n + 1
            '''

                elif len(shape) == 1: 
                    code += f'''
            {assign_to} = condition(input_n : input_n + {p(size)} - 1)
            input_n = input_n + {p(size)}
            '''

                else:
                    code += f'''
            {assign_to} = reshape(condition(input_n : input_n + {p(size)} - 1), {p(shape)})
            input_n = input_n + {p(size)}
            '''


        code += '''
        
            ! Allocate for rest of variables
            '''

        for v, ranges in self._context.variables.items():
            if v in initialized:
                continue
            if ranges:
                name = printer.fortran_name(v.name)
                ranges_str = ",".join(f"{p(r.start)}:{p(r.end)}" for r in ranges)
                code += f'''
            allocate({name}({ranges_str}))
            {name} = ieee_value({name}, ieee_quiet_nan)
            '''


        code += f'''

            ! Allocate for functions'''
            
        for name, ranges in self._context.functions.items():
            if ranges:
                f_name = printer.fortran_name(name)
                ranges_str = ",".join(f"{p(r.start)}:{p(r.end)}" for r in ranges)
                code += f'''
            allocate({f_name}({ranges_str}))
            {f_name} = ieee_value({f_name}, ieee_quiet_nan)
            '''

        code += f'''

            ! Allocate for Newton solver
            newton_size = 0'''  
        for size in self._context.newton_sizes:
            code += f'''
            newton_size = max(newton_size, {p(size)})'''

        code += f'''
            allocate(newton_eq(newton_size))
            allocate(newton_jac(newton_size, newton_size))
            allocate(newton_ipiv(newton_size))

            ! Main solver routine

            open(unit=log_unit, file=log_path, form='unformatted',&
                 access='stream', status='replace', iostat=ios)
            if (ios /= 0) then
               print *, "Error opening file: ", log_path
               stop 1
            end if

            error = 0
        '''

        for element in self._elements:
            code += textwrap.indent(element._generate(printer), '    ' * 3)
            code += '\n'

        code += f'''
        999 continue

            ! Write variables to log
        '''
        for v in self._context.variables:
            if v.indices:
                if v.space == space.Z:
                    code += f'''
            write(log_unit) real({printer.print_as_array_arg(v)}, kind=8)'''
                else:
                    code += f'''
            write(log_unit) {printer.print_as_array_arg(v)}'''
            else:
                if v.space == space.Z:
                    code += f'''
            write(log_unit) real({p(v)}, kind=8)'''
                else:
                    code += f'''
            write(log_unit) {p(v)}'''

        code += f'''
            ! Write functions to log
        '''
        for name in self._context.functions:
            code += f'''
            write(log_unit) {printer.fortran_name(name)}'''

        code += f'''

            call flush(6)
            close(log_unit)
        end subroutine run_solver
        '''

        code = textwrap.dedent(code)

        return code


def build_solver(check_nan: bool = True,
                 check_bounds: bool = True, 
                 check_dependencies: bool = True) -> SolverBuilder:
    return SolverBuilder(check_options={'nan': check_nan,
                                        'bounds': check_bounds,
                                        'dependencies': check_dependencies})