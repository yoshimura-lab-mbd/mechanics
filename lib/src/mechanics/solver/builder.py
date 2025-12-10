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
from .runner import ErrorReceiver, SolverRunner

class DependencyError(ValueError):
    pass

class SolverElement(ErrorReceiver):
    _root: 'SolverBuilder'

    _key_counter: int = 0

    def __init__(self, parent: Optional['SolverElement'] = None) -> None:
        super().__init__(parent)
        if parent is None:
            self._root = cast('SolverBuilder', self)
        else:
            self._root = parent._root

    def _generate(self, printer: FortranPrinter) -> str:
        return ''


class SolverBlock(SolverElement):
    _elements: list[SolverElement]

    def __init__(self, parent: Optional['SolverElement'] = None) -> None:
        super().__init__(parent)
        self._elements = []

    def explicit(self, equations: ExplicitEquations) -> Self:
        element = ExplicitElement(self._root, equations)
        self._elements.append(element)
        return self

    def calculate(self, functions: dict[str, Expr], *indices: Index) -> Self:
        element = CalculationElement(self._root, functions, *indices)
        self._elements.append(element)
        return self

    # @contextmanager
    # def implicit(self, equations: ImplicitEquations, unknowns: tuple_or_list[Variable],
    #              indices: tuple_or_list[tuple[Index, Expr, Expr]] = []):
    #     block = ImplicitBlock(self._root, equations, tuple(unknowns), list(indices))
    #     yield block
    #     self._elements.append(block)

    @contextmanager
    def steps(self, index: Index, start: Expr | int, end: Expr | int):
        block = StepsBlock(self._root, index, sympify(start), sympify(end))
        yield block
        self._elements.append(block)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, traceback):
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

    def __init__(self, parent: 'SolverElement', equations: ExplicitEquations) -> None:
        super().__init__(parent)
        self._equations = equations

        self._frame_info = inspect.stack()[2]

        for l, r in equations.items():
            if not isinstance(l, Variable):
                raise TypeError(f'Left-hand side of explicit equation must be a Variable, got {type(l)}: {l} = {r}.')

            for var in r.atoms(Variable):
                if var.general_form() not in self._root._constants and \
                   var.general_form() not in self._root._variables:
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
                if var in self._root._constants: continue
                if var in checked_dependencies: continue

                checks.append(var)
                checked_dependencies.add(var)

                bound_condition = self._root._bound_condition_of(var)
                if bound_condition == sp.S.true:
                    pass
                elif bound_condition == sp.S.false:
                    raise IndexError(f'Variable {var} out of bounds in explicit equation: {l} = {r}')
                else:
                    bound_error_key = self.register_error(self._bound_error, self._frame_info, var, l, r)

                if self._root.check_options.get('bounds', True):
                    code += f'''
        if ({p(sp.Not(bound_condition))}) then; error = {bound_error_key}; goto 999; end if'''
                
                if self._root.check_options.get('dependencies', True):
                    dependency_error_key = self.register_error(self._dependency_error, self._frame_info, var, l, r)
                    code += f'''
        if (ieee_is_nan({p(var)})) then; error = {dependency_error_key}; goto 999; end if
        '''


            code += f'''
        {p(l)} = {p(r)}'''

            if self._root.check_options.get('nan', True):
                nan_error_key = self.register_error(self._nan_error, self._frame_info, l, r)
                code += f'''
        if (ieee_is_nan({p(l)})) then; error = {nan_error_key}; goto 999; end if
        '''

        return textwrap.dedent(code)


class CalculationElement(SolverBlock):
    _functions: dict[str, Expr]
    _indices: tuple[Index, ...]

    def __init__(self, root: 'SolverBuilder', functions: dict[str, Expr], *indices: Index) -> None:
        super().__init__(root)
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

    def __init__(self, root: 'SolverBuilder', index: Index, start: Expr, end: Expr) -> None:
        super().__init__(root)
        self.index = index
        self.start = start
        self.end = end

        root._register_indices(index)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

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

class SolverBuilder(SolverBlock):
    _indices: set[Index]
    _variables: dict[Variable, tuple[IndexRange, ...]]
    _functions: dict[str, tuple[IndexRange, ...]]
    _constants: list[Variable]
    _inputs: list[Variable]

    _newton_sizes: list[Expr]

    check_options: dict[str, bool]

    def __init__(self, check_options: dict[str, bool] = {}) -> None:
        super().__init__()

        self._indices = set()
        self._variables = {}
        self._functions = {}
        self._constants = []
        self._inputs = []

        self._newton_sizes = []

        self.check_options = check_options

        SolverBlock._key_counter = 0

    def constants(self, *constants: Variable,
                  index: Optional[tuple[Index, Expr | int, Expr | int]] = None,
                  indices: list[tuple[Index, Expr | int, Expr | int]] = []) -> Self:
        self.variables(*constants, index=index, indices=indices)
        self._constants.extend(constants)
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
            self._variables[var] = index_ranges

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
            self._functions[name] = index_ranges

        return self

    def inputs(self, *inputs: Variable) -> Self:
        self._inputs.extend(inputs)
        return self


    def generate(self) -> SolverRunner:
        printer = FortranPrinter({'source_format': 'free', 'strict': False, 'standard': 95, 'precision': 15})
        code = self._generate(printer)
        return SolverRunner(
            code, self,
            constants = {c: self._ranges_of(c) for c in self._constants},
            variables = {v: self._ranges_of(v) for v in self._variables},
            inputs = {v: self._ranges_of(v) for v in self._inputs},
            functions = {name: self._ranges_of(name) for name in self._functions}
        )

    def _register_indices(self, *index: Index):
        self._indices.update(index)

    def _require_newton_size(self, size: Expr):
        self._newton_sizes.append(size)

    def _ranges_of(self, var: Variable | str) -> tuple[IndexRange, ...]:
        if isinstance(var, str):
            var_ranges = self._functions.get(var, None)
            if var_ranges is None:
                raise ValueError(f'Function {var} not defined in solver.')
            indices = {range.index: range.index for range in var_ranges}
        else:
            var_ranges = self._variables.get(var.general_form(), None)
            if var_ranges is None:
                raise ValueError(f'Variable {var} not defined in solver.')
            indices = var.index_subs
        
        ranges = []
        for range in var_ranges:
            if indices[range.index] == range.index:
                ranges.append(range)
        return tuple(ranges)

    def _shape_of(self, var: Variable | str, subs: Any = {}) -> tuple[Expr, ...]:
        shape = []
        for range in self._ranges_of(var):
            shape.append(sp.sympify(range.end - range.start + 1).subs(subs))
        return tuple(shape)

    def _size_of(self, var: Variable) -> Expr:
        size = sp.S.One
        for n in self._shape_of(var):
            size *= n
        return size

    def _shape_in_context_of(self, var: Variable, indices: list[tuple[Index, Expr, Expr]]) -> tuple[Expr, ...]:
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

    def _bound_condition_of(self, var: Variable) -> Expr:
        ranges = self._ranges_of(var)
        condition = sp.S.true
        for range in ranges:
            condition = condition & ((range.start <= range.index) & (range.index <= range.end))
        return condition

    def _unknowns_of(self, exprs: tuple_ish[Expr]) -> set[Variable]:
        unknowns = set()
        for expr in to_tuple(exprs):
            vars = sp.sympify(expr).atoms(Variable)
            for v in vars:
                if v not in self._constants and v not in self._inputs:
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

        for var in self._variables:
            type_ = 'integer' if var.space == space.Z else 'real(8)'
            if var.indices:
                code += f'''
            {type_}, allocatable :: {printer.print_as_array_arg(var)}'''
            else:
                code += f'''
            {type_} :: {p(var)}'''

        code += '''

            ! Define functions'''

        for name, indices in self._functions.items():
            if indices:
                code += f'''
            real(8), allocatable :: {printer.fortran_name(name)}({", ".join([":" for _ in indices])})'''
            else:
                code += f'''
            real(8) :: {printer.fortran_name(name)}'''

        code += '''

            ! Define indices'''
        for index in self._indices:
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
            integer :: newton_n

            ! Initialize variables
            input_n = 1
            '''
        
        initialized = set()

        for v in self._constants + self._inputs:
            var = v.general_form()
            ranges = self._variables.get(var, ())
            if var in self._variables and ranges and var not in initialized:
                initialized.add(var)
                name = printer.fortran_name(var.name)
                ranges_str = ",".join(f"{p(r.start)}:{p(r.end)}" for r in ranges)
                code += f'''
            allocate({name}({ranges_str}))
            {name} = ieee_value({name}, ieee_quiet_nan)'''

            name = printer.fortran_name(v.name)
            size = self._size_of(v)
            shape = self._shape_of(v)
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

        for v, ranges in self._variables.items():
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
            
        for name, ranges in self._functions.items():
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
        for size in self._newton_sizes:
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
        for v in self._variables:
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
        for name in self._functions:
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