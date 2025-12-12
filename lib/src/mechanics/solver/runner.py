from typing import Optional, Any, Callable
import os
import sys
import tempfile
import importlib.util
import subprocess
import numpy as np
import sympy as sp

from mechanics.symbol import Variable, Expr, Index, IndexRange
from .python import PythonPrinter

class ErrorReceiver:
    _parent: Optional['ErrorReceiver']
    _children: list['ErrorReceiver']
    _key_counter: int
    _callbacks: dict[int, tuple[Callable, tuple[Any, ...], dict[str, Any]]]

    def __init__(self, parent: Optional['ErrorReceiver'] = None) -> None:
        self._parent = parent
        self._children = []
        self._callbacks = {}

        if parent is None:
            self._key_counter = 0
        else:
            parent._children.append(self)

    def register_error(self, callback, *args, **kwargs) -> int:
        key = self._gen_key()
        self._callbacks[key] = (callback, args, kwargs)
        return key

    def _gen_key(self) -> int:
        if self._parent is None:
            self._key_counter += 1
            return self._key_counter
        else:
            return self._parent._gen_key()


    def receive_error(self, key: int) -> bool:
        return self._receive_error(key)

    def _receive_error(self, key: int) -> bool:
        if key in self._callbacks:
            callback, args, kwargs = self._callbacks[key]
            callback(*args, **kwargs)
            return True
        for child in self._children:
            if child._receive_error(key):
                return True
        return False
        
class SolverRunner:
    _generated: Any
    _receiver: ErrorReceiver
    _constants: dict[Variable, tuple[IndexRange, ...]]
    _variables: dict[Variable, tuple[IndexRange, ...]]
    _inputs: dict[Variable, tuple[IndexRange, ...]]
    _functions: dict[str, tuple[IndexRange, ...]]

    def __init__(self, source: str, receiver: ErrorReceiver,
                 constants: dict[Variable, tuple[IndexRange, ...]],
                 variables: dict[Variable, tuple[IndexRange, ...]],
                 inputs: dict[Variable, tuple[IndexRange, ...]],
                 functions: dict[str, tuple[IndexRange, ...]]):

        self._generated = self._compile_and_load(source)
        self._receiver = receiver

        self._constants = constants
        self._variables = variables
        self._inputs = inputs
        self._functions = functions


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

    def eval(self, expr: Expr | float | np.ndarray, condition: dict[Variable | Index, Any]) -> float | np.ndarray:
        if isinstance(expr, (float, np.ndarray)):
            return expr
        # print('Evaluating expression:', expr, 'with condition:', condition)
        f = sp.lambdify(list(condition.keys()), expr, printer=PythonPrinter())
        return f(*[v for v in condition.values()])

    def run(self, condition: dict[Variable, Expr | float | np.ndarray]) -> dict[Variable | str, float | np.ndarray]:

        condition_values = {}
        condition_evaluated: dict[Variable | Index, float | np.ndarray] = {}
        
        for c, value in condition.items():
            if c in self._constants:
                ranges = self._constants[c]

                shape = tuple(int(self.eval(r.end - r.start + 1, condition_evaluated)) for r in ranges)

                range_values = {r.index: np.arange(
                    int(self.eval(r.start, condition_evaluated)), 
                    int(self.eval(r.end, condition_evaluated)) + 1
                ) for r in ranges}

                c_evaluated = self.eval(condition[c], condition_evaluated | range_values)
                shape_given = np.shape(c_evaluated)
                if shape != shape_given:
                    raise ValueError(f'Constant {c} has shape {shape}, but got {shape_given}.')
                if np.ndim(c_evaluated) == 0:
                    condition_values[c] = [c_evaluated]
                    condition_evaluated[c] = c_evaluated
                else:
                    condition_values[c] = list(float(v) for v in np.ravel(c_evaluated, order='F'))
                    condition_evaluated[c] = c_evaluated

            elif c in self._inputs:
                ranges = self._inputs[c]

                shape = tuple(int(self.eval(r.end - r.start + 1, condition_evaluated)) for r in ranges)

                range_values = {r.index: np.arange(
                    int(self.eval(r.start, condition_evaluated)), 
                    int(self.eval(r.end, condition_evaluated)) + 1
                ) for r in ranges}

                v_evaluated = self.eval(condition[c], condition_evaluated | range_values)
                shape_given = np.shape(v_evaluated)

                if shape != shape_given:
                    raise ValueError(f'Input variable {c} has shape {shape}, but got {shape_given}.')
                if np.ndim(v_evaluated) == 0:
                    condition_values[c] = [v_evaluated]
                else:
                    condition_values[c] = list(v for v in np.ravel(v_evaluated, order='F'))
            else:
                raise ValueError(f'Condition for variable {c} is not provided as constant or input.')

        print(condition_values)

        condition_raw = []
        for c in self._constants.keys():
            condition_raw.extend(condition_values[c])
        for v in self._inputs.keys():
            condition_raw.extend(condition_values[v])

        result_dir = tempfile.mkdtemp()
        log_path = os.path.join(result_dir, 'result.log')

        error = self._generated.run_solver(log_path, condition_raw)
        if error != 0:
            # print(error)
            if not self._receiver.receive_error(error):
                raise RuntimeError(f'Solver execution failed with unknown error. Error code: {error}')

        log_data = np.memmap(log_path, dtype=np.float64, mode='r')

        print(f'Log data in: {log_path}, size={log_data.size}')

        result = {}
        for c, shape in self._constants.items():
            result[c] = condition[c]

        offset = 0
        for v, ranges in self._variables.items():
            if ranges == ():
                # print(v, offset, log_data[offset])
                result[v] = log_data[offset]
                offset += 1
            else:
                shape = tuple(int(self.eval(r.end - r.start + 1, condition_evaluated)) for r in ranges)
                size = np.prod(shape)
                # print(v, shape, size, offset)
                data = log_data[offset:offset + size].reshape(shape, order='F')
                result[v] = data
                offset += size

        for name, ranges in self._functions.items():
            if ranges == ():
                # print(name, offset, log_data[offset])
                result[name] = log_data[offset]
                offset += 1
            else:
                shape = tuple(int(self.eval(r.end - r.start + 1, condition_evaluated)) for r in ranges)
                size = np.prod(shape)
                data = log_data[offset:offset + size].reshape(shape, order='F')
                result[name] = data
                offset += size

        return result
