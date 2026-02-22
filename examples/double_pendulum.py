import marimo

__generated_with = "0.20.1"
app = marimo.App(width="medium")


@app.cell
def intro():
    import marimo as _mo

    _mo.md(r"""
    # Double pendulum (marimo)

    `mechanics` で導出した運動方程式を RK4 で数値積分し、軌道と時系列を描画します。
    """)
    return


@app.cell
def derive_lagrangian():
    import mechanics.space as space
    from mechanics import base_spaces, constants, variables
    from sympy import cos, diff, sin

    t, = base_spaces("t")

    def dot(f):
        return diff(f, t)

    theta1, theta2 = q = variables(r"\theta_1 \theta_2", t, space=space.S)
    dq = tuple(dot(q_n) for q_n in q)
    ddq = tuple(dot(dq_n) for dq_n in dq)

    g, m1, m2, l1, l2 = constants(r"g, m_1 m_2 \ell_1 \ell_2")

    x1 = l1 * sin(theta1)
    y1 = -l1 * cos(theta1)
    x2 = x1 + l2 * sin(theta2)
    y2 = y1 - l2 * cos(theta2)

    U = m1 * g * y1 + m2 * g * y2
    kinetic = (
        m1 / 2 * (dot(x1) ** 2 + dot(y1) ** 2)
        + m2 / 2 * (dot(x2) ** 2 + dot(y2) ** 2)
    ).simplify()
    E = kinetic + U
    L = kinetic - U
    return E, L, ddq, dq, g, l1, l2, m1, m2, q, x1, x2, y1, y2


@app.cell
def derive_equations(L, ddq, q):
    from mechanics.lagrange import euler_lagrange_equation
    from sympy import solve

    EL = euler_lagrange_equation(L, q)
    F = solve(EL, ddq)
    return EL, F


@app.cell(hide_code=True)
def show_symbolic_results(EL, F, L):
    import marimo as _mo
    from sympy import latex as _latex

    el_lines = "\n".join(f"$$ {_latex(eq)} $$" for eq in EL)
    f_lines = "\n".join(f"$$ {_latex(lhs)} = {_latex(rhs)} $$" for lhs, rhs in F.items())
    body = _mo.md(
        "\n".join(
            [
                "## Symbolic equations",
                "",
                "### Lagrangian",
                f"$$ L = {_latex(L)} $$",
                "",
                "### Euler-Lagrange equations",
                el_lines,
                "",
                "### Solved accelerations",
                f_lines,
            ]
        )
    ).style(
        {
            "overflow-x": "auto",
            "overflow-y": "hidden",
            "max-width": "100%",
            "padding-bottom": "0.25rem",
        }
    )
    body
    return


@app.cell
def build_step_equations(F, dq, q):
    from mechanics import constants as _constants, indices as _indices
    from mechanics.integrator.runge_kutta import rk4_explicit as _rk4_explicit

    h, total_time = _constants("h T")
    i, = _indices("i")
    X, K, step_eq, d = _rk4_explicit(F, h, i)

    theta1_, theta2_ = d(q)
    v_theta1_, v_theta2_ = d(dq)
    return (
        K,
        X,
        d,
        h,
        i,
        step_eq,
        theta1_,
        theta2_,
        total_time,
        v_theta1_,
        v_theta2_,
    )


@app.cell
def build_solver_module(
    E,
    K,
    X,
    d,
    g,
    h,
    i,
    l1,
    l2,
    m1,
    m2,
    step_eq,
    total_time,
    x1,
    x2,
    y1,
    y2,
):
    from mechanics.solver import build_solver

    solver = build_solver()
    solver.constants(g, m1, m2, l1, l2)
    solver.constants(h, total_time)
    solver.variables(*X, *K, index=(i, 0, total_time / h))
    solver.functions("x1", "y1", "x2", "y2", "E", index=(i, 0, total_time / h))
    solver.inputs(*(x[0] for x in X))
    with solver.steps(i, 0, total_time / h) as step:
        step.explicit(step_eq)
        step.calculate({"x1": d(x1), "y1": d(y1), "x2": d(x2), "y2": d(y2), "E": d(E)}, i)
    solver = solver.generate()
    return (solver,)


@app.cell
def controls():
    import marimo as _mo

    mass1 = _mo.ui.slider(0.1, 5.0, value=1.0, step=0.1, label="m1")
    mass2 = _mo.ui.slider(0.1, 5.0, value=1.0, step=0.1, label="m2")
    length1 = _mo.ui.slider(0.1, 3.0, value=1.0, step=0.1, label="l1")
    length2 = _mo.ui.slider(0.1, 3.0, value=1.0, step=0.1, label="l2")
    gravity = _mo.ui.slider(0.1, 20.0, value=9.8, step=0.1, label="g")

    theta1_0 = _mo.ui.slider(-3.14, 3.14, value=1.57, step=0.01, label="theta1(0)")
    theta2_0 = _mo.ui.slider(-3.14, 3.14, value=0.0, step=0.01, label="theta2(0)")
    v_theta1_0 = _mo.ui.slider(-5.0, 5.0, value=0.0, step=0.05, label="v_theta1(0)")
    v_theta2_0 = _mo.ui.slider(-5.0, 5.0, value=0.0, step=0.05, label="v_theta2(0)")

    step_size = _mo.ui.slider(0.001, 0.05, value=0.01, step=0.001, label="h")
    sim_time = _mo.ui.slider(1.0, 100.0, value=30.0, step=1.0, label="T")

    ui = _mo.vstack(
        [
            _mo.md("## Parameters"),
            _mo.hstack([mass1, mass2, length1, length2, gravity]),
            _mo.md("## Initial values"),
            _mo.hstack([theta1_0, theta2_0, v_theta1_0, v_theta2_0]),
            _mo.md("## Integrator"),
            _mo.hstack([step_size, sim_time]),
        ]
    )
    ui
    return (
        gravity,
        length1,
        length2,
        mass1,
        mass2,
        sim_time,
        step_size,
        theta1_0,
        theta2_0,
        v_theta1_0,
        v_theta2_0,
    )


@app.cell
def run_simulation(
    g,
    gravity,
    h,
    l1,
    l2,
    length1,
    length2,
    m1,
    m2,
    mass1,
    mass2,
    sim_time,
    solver,
    step_size,
    theta1_,
    theta1_0,
    theta2_,
    theta2_0,
    total_time,
    v_theta1_,
    v_theta1_0,
    v_theta2_,
    v_theta2_0,
):
    import numpy as np

    result = solver.run(
        {
            m1: mass1.value,
            m2: mass2.value,
            l1: length1.value,
            l2: length2.value,
            g: gravity.value,
            h: step_size.value,
            total_time: sim_time.value,
            theta1_[0]: theta1_0.value,
            theta2_[0]: theta2_0.value,
            v_theta1_[0]: v_theta1_0.value,
            v_theta2_[0]: v_theta2_0.value,
        }
    )
    t_ = np.arange(0, result[total_time], result[h])
    return result, t_


@app.cell
def plot_trajectory(result):
    import matplotlib.pyplot as _plt

    _plt.rcParams["font.family"] = "Times New Roman"
    _plt.rcParams["mathtext.fontset"] = "cm"

    _fig, _ax = _plt.subplots(1, 1, figsize=(4, 4), tight_layout=True)
    _ax.plot(result["x1"], result["y1"], label="$m_1$")
    _ax.plot(result["x2"], result["y2"], label="$m_2$")
    _ax.set_xlabel("$x$")
    _ax.set_ylabel("$y$")
    _ax.set_aspect("equal")
    _ax.legend()
    _fig
    return


@app.cell
def plot_timeseries(result, t_, theta1_, theta2_, v_theta1_, v_theta2_):
    import matplotlib.pyplot as _plt

    _fig, axes = _plt.subplots(3, 1, figsize=(6, 8), tight_layout=True)

    series = [
        (r"\theta", (theta1_, theta2_)),
        (r"v_\theta", (v_theta1_, v_theta2_)),
        ("E", ("E",)),
    ]
    for ax, (name, vars_) in zip(axes.flatten(), series):
        for var in vars_:
            label_name = getattr(var, "name", var)
            # ax.plot(t_, result[var][:-1], label=f"${label_name}$")
            ax.plot(t_, result[var][:-1])
        ax.set_xlabel("$t$")
        ax.set_ylabel(f"${name}$")
        ax.set_xlim(0, t_[-1] if len(t_) else 0)
        if len(vars_) > 1:
            ax.legend()
    _fig
    return


if __name__ == "__main__":
    app.run()
