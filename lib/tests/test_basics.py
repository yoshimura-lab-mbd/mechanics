import numpy as np
import pytest

import mechanics.space as space
from mechanics import base_space, base_spaces, constant, constants, index, indices, variable, variables

from examples.double_pendulum import app as double_pendulum_app


@pytest.fixture(scope="module")
def dp():
    """double_pendulum ノートブックを app.run() で一括実行し、全変数を返す。"""
    _, defs = double_pendulum_app.run()
    return defs


def test_double_pendulum_energy_conservation(dp):
    """RK4 積分でエネルギー誤差が 1% 未満であること"""
    E = dp["result"]["E"]
    drift = abs(E[-1] - E[0]) / abs(E[0])
    assert drift < 0.01


def test_double_pendulum_trajectory_length(dp):
    """軌道の配列長が時間ステップ数と整合していること"""
    result, t_ = dp["result"], dp["t_"]
    n = len(t_)
    for key in ("x1", "y1", "x2", "y2", "E"):
        assert len(result[key]) == n + 1, f"result[{key!r}] の長さが期待値と異なる"


def test_double_pendulum_pendulum_length(dp):
    """振り子長が保存されること（デフォルト: ell_1 = ell_2 = 1.0）"""
    result = dp["result"]
    x1, y1 = np.array(result["x1"]), np.array(result["y1"])
    x2, y2 = np.array(result["x2"]), np.array(result["y2"])
    assert np.allclose(np.sqrt(x1**2 + y1**2), 1.0, atol=1e-6), "第1振り子長が保存されていない"
    assert np.allclose(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2), 1.0, atol=1e-6), "第2振り子長が保存されていない"


def test_symbol_factories_return_namespace_and_singulars():
    from sympy import Symbol

    s = base_spaces(r"t x")
    assert s.t.name == "t"
    assert s.x.name == "x"

    i = indices("i j")
    assert i.i.name == "i"
    assert i.j.name == "j"

    t = base_space("t")
    q = variables(r"\theta_1 \theta_2", t, space=space.S)
    assert q.theta_1.name == r"\theta_1"
    assert q.theta_2.name == r"\theta_2"

    c = constants(r"g m_1 \ell_1")
    assert c.g.name == "g"
    assert c.m_1.name == "m_1"
    assert c.ell_1.name == r"\ell_1"

    v = variable("q", t)
    k = constant("k")
    idx = index("n")
    assert v.name == "q"
    assert k.name == "k"
    assert idx.name == "n"
