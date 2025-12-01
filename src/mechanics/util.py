from typing import Union, TypeVar
from numbers import Number
from itertools import product, count
import string
import inspect
import linecache
import sympy as sp

T = TypeVar('T')

tuple_or_list = Union[tuple[T, ...], list[T]]
tuple_ish = Union[T, tuple[T, ...], list[T]]
single_or_tuple = Union[T, tuple[T, ...]]

def split_latex(s, delimiters=(' ', ','), brackets={'(': ')', '{': '}', '[': ']'}) -> list[str]:
    result = []
    buf = []
    stack = []  # 括弧の種類を管理

    for char in s:
        if char in brackets.keys():  # 開き括弧
            stack.append(brackets[char])
            buf.append(char)
        elif char in brackets.values():  # 閉じ括弧
            if stack and char == stack[-1]:
                stack.pop()
            buf.append(char)
        elif char in delimiters and not stack:  # トップレベルの区切り文字
            if buf:
                result.append(''.join(buf).strip())
                buf.clear()
        else:
            buf.append(char)

    # 最後に残った文字列
    if buf:
        result.append(''.join(buf).strip())

    return result

python_name_trans = str.maketrans(
    {'\\': '', '{': '', '}': '', ',': '', ' ': '_', '^': '_', '-': 'm', '\'': 'prime'})
    
def python_name(name: str) -> str:
    name = name.translate(python_name_trans)
    if name == 'lambda':
        name = 'lambda_'
    return name

def to_tuple(items: tuple_ish[T]) -> tuple[T, ...]:
    if isinstance(items, tuple):
        return items
    elif isinstance(items, list):
        return tuple(items)
    else:
        return (items,)
    
def to_single_or_tuple(items: tuple_ish[T]) -> single_or_tuple[T]:
    if isinstance(items, tuple):
        result = items
    elif isinstance(items, list):
        result = tuple(items)
    else:
        result = (items,)

    if len(result) == 1:
        return result[0]
    else:
        return result
        
def generate_prefixes():
    """a, b, ..., z, aa, ab, ..., zz, aaa, ..."""

    letters = string.ascii_uppercase
    for n in count(1):  # prefix length
        for comb in product(letters, repeat=n):
            yield ''.join(comb)



RED   = "\033[31m"
CYAN  = "\033[36m"
BOLD  = "\033[1m"
RESET = "\033[0m"

def format_frameinfo(fi: inspect.FrameInfo, cursor_col=None):
    filename = fi.filename
    lineno   = fi.lineno
    code     = linecache.getline(filename, lineno).rstrip()

    out = []
    out.append(f"{BOLD}{CYAN}File \"{filename}\", line {lineno}{RESET}")
    out.append(f"  {code}")

    if cursor_col is not None and cursor_col <= len(code):
        pointer = " " * (cursor_col + 2) + f"{RED}^{RESET}"
        out.append(pointer)

    return "\n".join(out)


def sympify(expr: sp.Expr | int) -> sp.Expr:
    return sp.sympify(expr) # type: ignore