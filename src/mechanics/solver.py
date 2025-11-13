from typing import Self, Optional, cast, Any
import os
import sys
import tempfile
import importlib.util
import importlib.resources
import shutil
import subprocess
import sympy.printing.fortran
import textwrap
import time

from mechanics.function import ExplicitEquations, Function, Expr, Index
from mechanics.util import python_name
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
        if isinstance(f, Function):
            if f.indices:
                indices = [cast(str, self.doprint(i)) for i in f.indices]
                return f'{self.fortran_name(f.name)}({", ".join(indices)})'
            else:
                return f'{self.fortran_name(f.name)}'
        else:
            return super()._print_Function(f)
        
    
    def print_as_array_arg(self, f: Function) -> str:
        if not f.indices:
            return f'{self.fortran_name(f.name)}'
        return f'{self.fortran_name(f.name)}({",".join([":"] * len(f.indices))})'


class SolverBlock:
    _root: 'Solver'
    _sub_blocks: list['SolverBlock']

    def __init__(self, root: 'Solver') -> None:
        self._root = root
        self._sub_blocks = []

    def solve_explicit(self, equations: ExplicitEquations)-> Self:
        block = ExplicitBlock(self._root, equations)
        self._sub_blocks.append(block)
        return self

    def calculate(self, functions: dict[str, Function], *indices: Index) -> Self:
        block = FunctionsBlock(self._root, functions, *indices)
        self._sub_blocks.append(block)
        return self

    def steps(self, index: Index, start: Expr, end: Expr) -> 'IterationBlock':
        block = IterationBlock(self._root, index, start, end)
        self._sub_blocks.append(block)
        return block

    def _generate(self, printer: FortranPrinter) -> str:
        return ''

    def _get_indices(self) -> set[Index]:
        indices = set()
        for block in self._sub_blocks:
            indices |= block._get_indices()
        return indices

class Solver(SolverBlock):
    _constants: list[Function]
    _variables: dict[Function, list[tuple[Index, Expr, Expr]]]
    _functions: dict[str, list[tuple[Index, Expr, Expr]]]
    _inputs: list[Function]

    _generated: Any
    _generate_dir: Optional[str]

    def __init__(self) -> None:
        super().__init__(self)
        self._constants = []
        self._variables = {}
        self._functions = {}
        self._inputs = []
        self._generate_dir = None
        
    def __del__(self):
        if self._generate_dir and os.path.exists(self._generate_dir):
            shutil.rmtree(self._generate_dir)

    def constants(self, *constants: Function) -> Self:
        self._constants.extend(constants)
        return self

    def variables(self, *variables: Function, 
                  index: Optional[tuple[Index, Expr, Expr]] = None,
                  indices: list[tuple[Index, Expr, Expr]] = []) -> Self:
        if index:
            indices = [index] + indices

        for i, start, end in indices:
            if not isinstance(i, Index):
                raise TypeError(f"Expected Index, got {i} which has type {type(i)})")

        for var in variables:
            self._variables[var] = indices

        return self

    def functions(self, *names: str, 
                  index: Optional[tuple[Index, Expr, Expr]] = None,
                  indices: list[tuple[Index, Expr, Expr]] = []) -> Self:
        if index:
            indices = [index] + indices
        
        for i, start, end in indices:
            if not isinstance(i, Index):
                raise TypeError(f"Expected Index, got {i} which has type {type(i)})")
        
        for name in names:
            self._functions[name] = indices

        return self

    def inputs(self, *inputs: Function) -> Self:
        self._inputs.extend(inputs)
        return self

    def run(self, condition: dict[Function, float]) -> None:
        if self._generated is None:
            raise RuntimeError('Solver has not been compiled. Use "with Solver() as solver:" syntax.')

        condition_values = []

        for c in self._constants:
            if c not in condition:
                raise ValueError(f'Constant {c} not provided in condition.')
            condition_values.append(float(condition[c]))

        for v in self._inputs:
            if v not in condition:
                raise ValueError(f'Input variable {v} not provided in condition.')
            condition_values.append(float(condition[v]))

        result_dir = tempfile.mkdtemp()
        log_path = os.path.join(result_dir, 'result.log')

        status, message = self._generated.run_solver(log_path, condition_values)
        if status != 0:
            time.sleep(0.1)
            raise RuntimeError(message.decode('utf-8'))


    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        printer = FortranPrinter({'source_format': 'free', 'strict': False, 'standard': 95, 'precision': 15})
        code = self._generate(printer)
        # print(code)

        self._generated = self._compile_and_load(code)


    def _generate(self, printer: FortranPrinter) -> str:
        p = printer.doprint

        code = ''
        code += f'''
        module constants
            real(8), save :: dummy_ = 0.0d0
        '''
        for c in self._constants:
            if c.space == space.Z:
                code += f'''
            integer, save :: {p(c)} = 0'''
            else: 
                code += f'''
            real(8), save :: {p(c)} = 0.0d0'''
        code += f'''
        end module constants

        subroutine run_solver(log_path, condition, status, message)
            use, intrinsic :: ieee_arithmetic
            use constants
            implicit none
            double precision dnrm2
            external dnrm2

            character(len=*), intent(in) :: log_path
            real(8), dimension(:), intent(in) :: condition
            integer, intent(out) :: status
            character(len=100), intent(out) :: message

            integer :: log_unit = 20
            integer :: ios
            
            ! Define variables'''

        for var in self._variables:
            if var.indices:
                code += f'''
            real(8), allocatable :: {printer.print_as_array_arg(var)}'''
            else:
                code += f'''
            real(8) :: {p(var)} = 0.0d0'''

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

            ! Initialize constants'''

        for n, c in enumerate(self._constants):
            if c.space == space.Z:
                code += f'''
            {p(c)} = int(condition({n + 1}))'''
            else: 
                code += f'''
            {p(c)} = condition({n + 1})'''
        code += '''
        
            ! Allocate variables'''

        for v, indices in self._variables.items():
            if indices:
                name = printer.fortran_name(v.name)
                ranges = ",".join(f"int({p(start)}):int({p(end)})" for i, start, end in indices)
                code += f'''
            allocate({name}({ranges}))'''
        code += '''

            ! Initialize variables using inputs'''

        for n, v in enumerate(self._inputs):
            code += f'''
            {p(v)} = condition({len(self._constants) + n + 1})'''
                
        code += f'''

            print *, "Started"
            print *, "Output in ", log_path

            open(unit=log_unit, file=log_path//"log.bin", form='unformatted',&
                 access='stream', status='replace', iostat=ios)
            if (ios /= 0) then
               print *, "Error opening file: ", log_path//"log.bin"
               stop 1
            end if

        '''

        for block in self._sub_blocks:
            code += textwrap.indent(block._generate(printer), '    ' * 3)
            code += '\n'

        code += f'''
            print *, "Completed"
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

class ExplicitBlock(SolverBlock):
    _equations: ExplicitEquations

    def __init__(self, root: 'Solver', equations: ExplicitEquations) -> None:
        super().__init__(root)
        self._equations = equations

    def _generate(self, printer: FortranPrinter) -> str:
        p = printer.doprint

        code = '\n! Solve explicit equations\n'
        for l, r in self._equations.items():
            code += f'{p(l)} = {p(r)}\n'

        return code


class FunctionsBlock(SolverBlock):
    _functions: dict[str, Function]
    _indices: tuple[Index, ...]

    def __init__(self, root: 'Solver', functions: dict[str, Function], *indices: Index) -> None:
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


class IterationBlock(SolverBlock):
    index: Index
    start: Expr
    end: Expr

    def __init__(self, root: 'Solver', index: Index, start: Expr, end: Expr) -> None:
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
            do {p(self.index)} = int({p(self.start)}), int({p(self.end)}) - 1 '''

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



