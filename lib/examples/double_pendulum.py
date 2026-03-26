import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell
def imports():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from sympy import cos, diff, sin, solve

    import mechanics.space as space
    from mechanics import Markdown, base_space, constants, index, variables
    from mechanics.differential_equation import to_first_order
    from mechanics.integrator.runge_kutta import rk4_explicit
    from mechanics.lagrange import euler_lagrange_equation
    from mechanics.solver import build_solver

    return (
        Markdown,
        base_space,
        build_solver,
        constants,
        cos,
        diff,
        euler_lagrange_equation,
        index,
        mo,
        np,
        plt,
        rk4_explicit,
        sin,
        solve,
        space,
        to_first_order,
        variables,
    )


@app.cell
def intro(mo):
    mo.md("""
    # Double pendulum
    """)
    return


@app.cell
def derive_lagrangian(base_space, constants, cos, diff, sin, space, variables):
    t = base_space("t")

    def dot(f):
        return diff(f, t)

    q = variables(r"\theta_1 \theta_2", t, space=space.S)
    theta1 = q.theta_1
    theta2 = q.theta_2

    dq = tuple(dot(q_n) for q_n in q)
    ddq = tuple(dot(dq_n) for dq_n in dq)

    c = constants(r"g, m_1 m_2 \ell_1 \ell_2")

    x1 = c.ell_1 * sin(theta1)
    y1 = -c.ell_1 * cos(theta1)
    x2 = x1 + c.ell_2 * sin(theta2)
    y2 = y1 - c.ell_2 * cos(theta2)

    U = c.m_1 * c.g * y1 + c.m_2 * c.g * y2
    kinetic = (
        c.m_1 / 2 * (dot(x1) ** 2 + dot(y1) ** 2)
        + c.m_2 / 2 * (dot(x2) ** 2 + dot(y2) ** 2)
    ).simplify()
    E = kinetic + U
    L = kinetic - U
    return E, L, c, ddq, q, x1, x2, y1, y2


@app.cell
def derive_equations(L, ddq, euler_lagrange_equation, q, solve):
    EL = euler_lagrange_equation(L, tuple(q))
    F = solve(EL, ddq)
    return EL, F


@app.cell(hide_code=True)
def show_symbolic_results(EL, F, L, Markdown, mo):
    md = Markdown()
    md.add_markdown("### Lagrangian")
    md.show("L = ", L)
    md.add_markdown("### Euler-Lagrange equations")
    md.show_equations(EL)
    md.add_markdown("### Solved accelerations")
    md.show_equations(F)
    md.render(mo)
    return


@app.cell
def build_step_equations(F, constants, index, q, rk4_explicit, to_first_order):
    ht = constants("h T")
    h = ht.h
    T = ht.T
    i = index("i")

    first_order = to_first_order(F)
    rk4 = rk4_explicit(first_order.equations, h, i)
    X = rk4.state_variables
    K = rk4.stage_variables
    step_eq = rk4.step_equations
    d = rk4.conversion * first_order.conversion
    q_ = d(q)
    v_ = d(first_order.variables_of_order(1))
    return K, T, X, d, h, i, q_, step_eq, v_


@app.cell
def build_solver_module(
    E,
    K,
    T,
    X,
    build_solver,
    c,
    d,
    h,
    i,
    step_eq,
    x1,
    x2,
    y1,
    y2,
):
    solver = build_solver()
    solver.constants(*c)
    solver.constants(h, T)
    solver.variables(*X, *K, index=(i, 0, T / h))
    solver.functions("x1", "y1", "x2", "y2", "E", index=(i, 0, T / h))
    solver.inputs(*(x[0] for x in X))
    with solver.steps(i, 0, T / h) as step:
        step.explicit(step_eq)
        step.calculate({"x1": d(x1), "y1": d(y1), "x2": d(x2), "y2": d(y2), "E": d(E)}, i)
    solver = solver.generate()
    return (solver,)


@app.cell
def controls(mo):
    sliders = mo.ui.dictionary(
        {
            "m_1": mo.ui.slider(start=0.1, stop=10.0, step=0.1, value=1.0, label=r"$m_1$"),
            "m_2": mo.ui.slider(start=0.1, stop=10.0, step=0.1, value=1.0, label=r"$m_2$"),
            "ell_1": mo.ui.slider(start=0.1, stop=5.0, step=0.1, value=1.0, label=r"$\ell_1$"),
            "ell_2": mo.ui.slider(start=0.1, stop=5.0, step=0.1, value=1.0, label=r"$\ell_2$"),
            "g": mo.ui.slider(start=0.0, stop=20.0, step=0.1, value=9.8, label=r"$g$"),
            "theta_1_0": mo.ui.slider(start=-3.14, stop=3.14, step=0.01, value=1.57, label=r"$\theta_1(0)$"),
            "theta_2_0": mo.ui.slider(start=-3.14, stop=3.14, step=0.01, value=0.0, label=r"$\theta_2(0)$"),
            "v_theta_1_0": mo.ui.slider(start=-10.0, stop=10.0, step=0.1, value=0.0, label=r"$\dot\theta_1(0)$"),
            "v_theta_2_0": mo.ui.slider(start=-10.0, stop=10.0, step=0.1, value=0.0, label=r"$\dot\theta_2(0)$"),
            "h": mo.ui.slider(start=0.001, stop=0.1, step=0.001, value=0.01, label=r"$h$"),
            "T": mo.ui.slider(start=1.0, stop=100.0, step=1.0, value=30.0, label=r"$T$"),
        }
    )
    sliders
    return (sliders,)


@app.cell
def run_simulation(T, c, h, np, q_, sliders, solver, v_):
    sv = sliders.value
    result = solver.run(
        {
            c.m_1: sv["m_1"],
            c.m_2: sv["m_2"],
            c.ell_1: sv["ell_1"],
            c.ell_2: sv["ell_2"],
            c.g: sv["g"],
            q_.theta_1[0]: sv["theta_1_0"],
            q_.theta_2[0]: sv["theta_2_0"],
            v_.theta_1[0]: sv["v_theta_1_0"],
            v_.theta_2[0]: sv["v_theta_2_0"],
            h: sv["h"],
            T: sv["T"],
        }
    )
    t_ = np.arange(0, result[T], result[h])
    return result, t_


@app.cell
def plot_trajectory(plt, result):
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "cm"

    _fig, _ax = plt.subplots(1, 1, figsize=(4, 4), tight_layout=True)
    _ax.plot(result["x1"], result["y1"], label="$m_1$")
    _ax.plot(result["x2"], result["y2"], label="$m_2$")
    _ax.set_xlabel("$x$")
    _ax.set_ylabel("$y$")
    _ax.set_aspect("equal")
    _ax.legend()
    return


@app.cell
def plot_timeseries(plt, q_, result, t_, v_):
    _fig, axes = plt.subplots(3, 1, figsize=(6, 8), tight_layout=True)

    series = [
        (r"\theta", (q_.theta_1, q_.theta_2)),
        (r"v_\theta", (v_.theta_1, v_.theta_2)),
        ("E", ("E",)),
    ]
    for ax, (name, vars_) in zip(axes.flatten(), series):
        for var in vars_:
            ax.plot(t_, result[var][:-1])
        ax.set_xlabel("$t$")
        ax.set_ylabel(f"${name}$")
        ax.set_xlim(0, t_[-1] if len(t_) else 0)
    _fig
    return


if __name__ == "__main__":
    app.run()
