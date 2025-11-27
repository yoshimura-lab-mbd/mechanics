from typing import Self, Optional, cast, Any
import os
import sys
import tempfile
import importlib.util
from contextlib import contextmanager
import shutil
import subprocess
import sympy.printing.fortran
import textwrap
from collections import defaultdict
import inspect
import numpy as np
import sympy as sp

from mechanics.symbol import ExplicitEquations, Variable, Expr, Index
from mechanics.util import format_frameinfo, python_name, to_tuple, tuple_ish, format_frameinfo
import mechanics.space as space

class FortranPrinter(sympy.printing.fortran.FCodePrinter):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.prefixes: dict[str, str] = {}

    def fortran_name(self, name: str) -> str:
        name = python_name(name)
        name = ''.join((c.lower() + '_') if c.isupper() else c for c in name)
        return name

    def _print_Symbol(self, symbol):
        if isinstance(symbol, Index):
            return f'{self.fortran_name(symbol.name)}'
        else:
            str(symbol)
    
    def _print_Function(self, f):
        if isinstance(f, Variable):
            if f.indices:
                indices = [cast(str, self.doprint(i)) for i in f.indices]
                return f'{self.fortran_name(f.name)}({", ".join(indices)})'
            else:
                return f'{self.fortran_name(f.name)}'
        else:
            return super()._print_Function(f)
        
    
    def print_as_array_arg(self, f: Variable) -> str:
        if not f.indices:
            return f'{self.fortran_name(f.name)}'
        args = []
        for i, value in f.index_subs.items():
            if i == value:
                args.append(':')
            else:
                args.append(f'{self.doprint(value)}')

        return f'{self.fortran_name(f.name)}({",".join(args)})'


class SolverBlock:
    _root: 'Solver'

    _key_counter: int = 0

    def __init__(self, root: 'Solver') -> None:
        self._root = root

    def _generate(self, printer: FortranPrinter) -> str:
        return ''

    def _get_indices(self) -> set[Index]:
        return set()

    def _gen_key(self) -> int:
        SolverBlock._key_counter += 1
        return SolverBlock._key_counter

    def _receive_error(self, key: int):
        pass

class NestBlock(SolverBlock):
    _sub_blocks: list[SolverBlock]

    def __init__(self, root: 'Solver') -> None:
        super().__init__(root)
        self._sub_blocks = []

    def _generate(self, printer: FortranPrinter) -> str:
        code = ''
        for block in self._sub_blocks:
            code += block._generate(printer)
            code += '\n'
        return code

    def _get_indices(self) -> set[Index]:
        indices = set()
        for block in self._sub_blocks:
            indices |= block._get_indices()
        return indices

    def _receive_error(self, key: int):
        for block in self._sub_blocks:
            block._receive_error(key)

class ExplicitBlock(SolverBlock):
    _equations: ExplicitEquations

    _bounds_checks: defaultdict[Variable, list[int]]
    _bounds_sources: dict[int, tuple[Variable, Variable, Expr]]
    _dependency_checks: defaultdict[Variable, list[int]]
    _dependency_sources: dict[int, tuple[Variable, Variable, Expr]]
    _frame_info: inspect.FrameInfo

    def __init__(self, root: 'Solver', equations: ExplicitEquations) -> None:
        super().__init__(root)
        self._equations = equations

        self._bounds_checks = defaultdict(list)
        self._bounds_sources = {}
        self._dependency_checks = defaultdict(list)
        self._dependency_sources = {}

        checked_dependencies = set()

        self._frame_info = inspect.stack()[2]

        for l, r in equations.items():
            if not isinstance(l, Variable):
                raise TypeError(f'Left-hand side of explicit equation must be a Variable, got {type(l)}: {l} = {r}.')
            dependencies = r.atoms(Variable) - checked_dependencies
            checks = []
            for var in dependencies:
                if var.general_form() in self._root._constants:
                    pass 
                elif var.general_form() in self._root._variables:
                    checks.append(var)
                else:
                    raise ValueError(f'Variable {var} used before definition in explicit equation: {l} = {r}.')

            for check in checks:
                key = self._gen_key()
                self._bounds_checks[l].append(key)
                self._bounds_sources[key] = (check, l, r)
                key = self._gen_key()
                self._dependency_checks[l].append(key)
                self._dependency_sources[key] = (check, l, r)

            checked_dependencies |= dependencies


    def _generate(self, printer: FortranPrinter) -> str:
        p = printer.doprint

        code = '\n! Solve explicit equations\n'

        if self._root._bounds_check:
            for l, r in self._equations.items():
                for key in self._bounds_checks[l]:
                    var, _, _ = self._bounds_sources[key]
                    ranges = self._root._range_of(var.general_form())
                    condition = ' .or. '.join(
                        f'{p(i)} < {p(lower)} .or. {p(i)} > {p(upper)}' 
                        for (_, lower, upper), i in zip(ranges, var.indices))
                    code += f'if ({condition}) error = {key}\n'

        for l, r in self._equations.items():
            if self._root._dependency_check:
                for key in self._dependency_checks[l]:
                    var, _, _ = self._dependency_sources[key]
                    code += f'if (ieee_is_nan({p(var)})) error = {key}\n'

            code += f'{p(l)} = {p(r)}\n'

        code += 'if (error /= 0) goto 999\n'

        return code

    def _receive_error(self, key: int):
        if key in self._dependency_sources:
            var, l, r = self._dependency_sources[key]
            raise DependencyError(
                f'Variable {var} used before calculation in explicit equation: {l} = {r}'
                + '\n at ' + format_frameinfo(self._frame_info)
                )
        elif key in self._bounds_sources:
            var, l, r = self._bounds_sources[key]
            raise IndexError(
                f'Variable {var} out of bounds in explicit equation: {l} = {r}'
                + '\n at ' + format_frameinfo(self._frame_info)
                )


class FunctionsBlock(SolverBlock):
    _functions: dict[str, Expr]
    _indices: tuple[Index, ...]

    def __init__(self, root: 'Solver', functions: dict[str, Expr], *indices: Index) -> None:
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


class IterationBlock(NestBlock):
    index: Index
    start: Expr | int
    end: Expr | int

    def __init__(self, root: 'Solver', index: Index, start: Expr | int, end: Expr | int) -> None:
        super().__init__(root)
        self.index = index
        self.start = start
        self.end = end

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def _generate(self, printer: FortranPrinter) -> str:
        p = printer.doprint

        code = f'''
            ! Iterate over {p(self.index)}
            do {p(self.index)} = {p(self.start)}, {p(self.end)} - 1 '''

        for block in self._sub_blocks:
            code += '\n'
            code += textwrap.indent(block._generate(printer), '    ' * 4)

        code += f'''
            end do
        '''

        return textwrap.dedent(code)

    def _get_indices(self) -> set[Index]:
        indices = super()._get_indices()
        indices.add(self.index)
        return indices




class Solver(NestBlock):
    _variables: dict[Variable, list[tuple[Index, Expr, Expr]]]
    _functions: dict[str, list[tuple[Index, Expr, Expr]]]
    _constants: list[Variable]
    _inputs: list[Variable]

    _context: NestBlock

    _generated: Any
    _generate_dir: Optional[str]

    _bounds_check: bool
    _dependency_check: bool

    def __init__(self, 
                 bounds_check: bool = True, 
                 dependency_check: bool = True) -> None:
        super().__init__(self)
        self._variables = {}
        self._functions = {}
        self._constants = []
        self._inputs = []

        self._context = self

        self._bounds_check = bounds_check
        self._dependency_check = dependency_check

        self._generate_dir = None

        SolverBlock._key_counter = 0
        
    def __del__(self):
        if self._generate_dir and os.path.exists(self._generate_dir):
            shutil.rmtree(self._generate_dir)

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

        for i, start, end in indices:
            if not isinstance(i, Index):
                raise TypeError(f"Expected Index, got {i} which has type {type(i)})")

        for var in variables:
            unknowns = self._unknowns(var.args)
            if unknowns:
                raise ValueError(f'{var} contains unknown variables or constants: {unknowns}. '
                                  'Add them using .variables() or .constants() first.')
            for index in indices:
                unknowns = self._unknowns(index) #type:ignore
                if unknowns:
                    raise ValueError(f'Index specification {index} contains unknown constants: {unknowns}. '
                                      'Add them using .constants() first.')
            self._variables[var] = tuple(sp.sympify(i) for i in indices) #type: ignore

        return self

    def functions(self, *names: str, 
                  index: Optional[tuple[Index, Expr | int, Expr | int]] = None,
                  indices: list[tuple[Index, Expr | int, Expr | int]] = []) -> Self:
        if index:
            indices = [index] + indices
        
        for i, start, end in indices:
            if not isinstance(i, Index):
                raise TypeError(f"Expected Index, got {i} which has type {type(i)})")
        
        for name in names:
            self._functions[name] = tuple(sp.sympify(i) for i in indices) #type: ignore

        return self

    def inputs(self, *inputs: Variable) -> Self:
        self._inputs.extend(inputs)
        return self

    def explicit(self, equations: ExplicitEquations) -> Self:
        block = ExplicitBlock(self._root, equations)
        self._context._sub_blocks.append(block)
        return self

    def calculate(self, functions: dict[str, Expr], *indices: Index) -> Self:
        block = FunctionsBlock(self._root, functions, *indices)
        self._context._sub_blocks.append(block)
        return self

    @contextmanager
    def steps(self, index: Index, start: Expr | int, end: Expr | int):
        block = IterationBlock(self._root, index, start, end)
        self._context._sub_blocks.append(block)
        context_back = self._context
        self._context = block
        yield
        self._context = context_back

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        printer = FortranPrinter({'source_format': 'free', 'strict': False, 'standard': 95, 'precision': 15})
        code = self._generate(printer)
        # print(code)

        self._generated = self._compile_and_load(code)

    def _range_of(self, var: Variable) -> tuple[tuple[Index, Expr, Expr], ...]:
        var_shape = self._variables.get(var.general_form(), None)
        if var_shape is None:
            raise ValueError(f'Variable {var} not defined in solver.')
        return tuple(var_shape)

    def _shape_of(self, var: Variable, subs: Any = {}) -> tuple[Expr, ...]:
        shape = []
        var_shape = self._variables.get(var.general_form(), None)
        if var_shape is None:
            raise ValueError(f'Variable {var} not defined in solver.')
        
        indices = var.index_subs
        for i, start, end in var_shape:
            if indices[i] == i:
                shape.append(sp.sympify(end - start + 1).subs(subs))
        return tuple(shape)

    def _size_of(self, var: Variable) -> Expr:
        size = sp.S.One
        for n in self._shape_of(var):
            size *= n
        return size

    def _unknowns(self, exprs: tuple_ish[Expr]) -> set[Variable]:
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
            real(8) :: {printer.fortran_name(name)} = 0.0d0'''

        code += '''

            ! Define indices'''
        for index in self._get_indices():
            code += f'''
            integer :: {p(index)} = 0'''

        code += '''

            ! Initialize variables
            input_n = 1
            '''
        
        initialized = set()

        for v in self._constants + self._inputs:
            var = v.general_form()
            indices = self._variables.get(var, [])
            if var in self._variables and indices and var not in initialized:
                initialized.add(var)
                name = printer.fortran_name(var.name)
                ranges = ",".join(f"{p(start)}:{p(end)}" for i, start, end in indices)
                code += f'''
            allocate({name}({ranges}))
            {name} = ieee_value({name}, ieee_quiet_nan)'''

            name = printer.fortran_name(v.name)
            size = self._size_of(v)
            shape = self._shape_of(v)
            assign_to = printer.print_as_array_arg(v) if var != v else name

            if v.space == space.Z:
                if len(shape) == 0:
                    code += f'''
            {assign_to} = int(condition(input_n))
            input_n = input_n + 1'''

                elif len(shape) == 1: 
                    code += f'''
            {assign_to} = int(condition(input_n : input_n + {p(size)} - 1))
            input_n = input_n + {p(size)}'''

                else:
                    code += f'''
            {assign_to} = reshape(int(condition(input_n : input_n + {p(size)} - 1)), {p(shape)})
            input_n = input_n + {p(size)}'''


            else:
                if len(shape) == 0:
                    code += f'''
            {assign_to} = condition(input_n)
            input_n = input_n + 1'''

                elif len(shape) == 1: 
                    code += f'''
            {assign_to} = condition(input_n : input_n + {p(size)} - 1)
            input_n = input_n + {p(size)}'''

                else:
                    code += f'''
            {assign_to} = reshape(condition(input_n : input_n + {p(size)} - 1), {p(shape)})
            input_n = input_n + {p(size)}'''


        code += '''
        
            ! Allocate for rest of variables
            '''

        for v, indices in self._variables.items():
            if v in initialized:
                continue
            if indices:
                name = printer.fortran_name(v.name)
                ranges = ",".join(f"{p(start)}:{p(end)}" for i, start, end in indices)
                code += f'''
            allocate({name}({ranges}))
            {name} = ieee_value({name}, ieee_quiet_nan)
            '''


        code += f'''

            ! Allocate for functions'''
            
        for name, indices in self._functions.items():
            if indices:
                f_name = printer.fortran_name(name)
                ranges = ",".join(f"{p(start)}:{p(end)}" for i, start, end in indices)
                code += f'''
            allocate({f_name}({ranges}))
            {f_name} = ieee_value({f_name}, ieee_quiet_nan)'''
                
        code += f'''

            open(unit=log_unit, file=log_path, form='unformatted',&
                 access='stream', status='replace', iostat=ios)
            if (ios /= 0) then
               print *, "Error opening file: ", log_path
               stop 1
            end if

            error = 0
        '''

        for block in self._sub_blocks:
            code += textwrap.indent(block._generate(printer), '    ' * 3)
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
    
    def _compile_and_load(self, source: str, libs: list[str] = []) -> Any:

        self._generate_dir = tempfile.mkdtemp()
        generate_path = os.path.join(self._generate_dir, 'generated.f90')

        with open(generate_path, 'w') as f:
            f.write(source)
        print(f'Generating Fortran code in {generate_path}')

        # lib_files = [str(importlib.resources.files('mechanics').joinpath(f'fortran/{filename}'))
        #     for filename in libs]
        lib_files = []

        shell_path = subprocess.check_output(['bash', '-l', '-c', 'echo $PATH']).decode().strip()
        env = os.environ.copy()
        env["PATH"] += ':' + shell_path
        generated_name = 'generated'
        ret = subprocess.run([
            sys.executable, '-m', 'numpy.f2py', '-m', generated_name, 
            '-c', 'generated.f90'] + lib_files 
            + ['--build-dir', 'build', '--f90flags="-Wno-unused-dummy-argument"']
            ,
            env=env, cwd=self._generate_dir,
            stdout=subprocess.PIPE,
            # stderr=subprocess.PIPE 
            # stdout=subprocess.DEVNULL,
            # stderr=sys.stderr
        )
        if ret.returncode != 0:
            print('======================== Compilation failed ========================')
            print(ret.stdout.decode())
            raise RuntimeError(f'Compilation failed with return code {ret.returncode}')

        so_file = next(p for p in os.listdir(self._generate_dir) if p.startswith(generated_name) and p.endswith('.so'))
        so_fullpath = os.path.join(self._generate_dir, so_file)

        spec = importlib.util.spec_from_file_location(generated_name, so_fullpath)
        if spec is None or spec.loader is None:
            raise ImportError(f'Could not load compiled module from {so_fullpath}')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        return module

    def run(self, condition: dict[Variable, Any]) -> dict[Variable | str, float | np.ndarray]:
        if self._generated is None:
            raise RuntimeError('Solver has not been compiled. Use "with Solver() as solver:" syntax.')

        condition_values = []
        condition_evaled: dict[Variable, Any] = {}

        for c in self._constants:
            if c not in condition:
                raise ValueError(f'Constant {c} not provided in condition.')
            if self._shape_of(c, condition_evaled) != np.shape(condition[c]):
                raise ValueError(f'Constant {c} has shape {self._shape_of(c, condition_evaled)}, but got {np.shape(condition[c])}.')
            if np.ndim(condition[c]) == 0:
                condition_values.append(float(condition[c]))
                condition_evaled[c] = condition[c]
            else:
                condition_values.extend(float(v) for v in np.ravel(condition[c], order='F'))
                condition_evaled[c] = condition[c]

        for v in self._inputs:
            if v not in condition:
                raise ValueError(f'Input variable {v} not provided in condition.')
            if self._shape_of(v, condition_evaled) != np.shape(condition[v]):
                raise ValueError(f'Input variable {v} has shape {self._shape_of(v, condition_evaled)}, but got {np.shape(condition[v])}.')
            if np.ndim(condition[v]) == 0:
                condition_values.append(float(condition[v]))
            else:
                condition_values.extend(float(v) for v in np.ravel(condition[v], order='F'))

        result_dir = tempfile.mkdtemp()
        log_path = os.path.join(result_dir, 'result.log')

        error = self._generated.run_solver(log_path, condition_values)
        if error != 0:
            self._receive_error(error)
            # raise RuntimeError(f'Solver execution failed with error code {error}.')

        log_data = np.memmap(log_path, dtype=np.float64, mode='r')

        print(f'Log data in: {log_path}, size={log_data.size}')

        result = {}
        shape_constants = {}
        for c in self._constants:
            result[c] = condition[c]
            if self._shape_of(c) == ():
                shape_constants[c] = condition[c]

        offset = 0
        for v, indices in self._variables.items():
            if indices:
                shape = tuple(int(cast(Expr, sp.sympify(end - start + 1)).subs(shape_constants)) 
                              for i, start, end in indices)
                size = np.prod(shape)
                # print(v, shape, size, offset)
                data = log_data[offset:offset + size].reshape(shape, order='F')
                result[v] = data
                offset += size
            else:
                # print(v, offset, log_data[offset])
                result[v] = log_data[offset]
                offset += 1

        for name, indices in self._functions.items():
            if indices:
                shape = tuple(int(cast(Expr, sp.sympify(end - start + 1)).subs(shape_constants)) 
                              for i, start, end in indices)
                size = np.prod(shape)
                data = log_data[offset:offset + size].reshape(shape, order='F')
                result[name] = data
                offset += size
            else:
                result[name] = log_data[offset]
                offset += 1

        return result


class DependencyError(ValueError):
    pass