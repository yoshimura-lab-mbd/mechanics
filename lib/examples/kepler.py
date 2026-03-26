import marimo

__generated_with = "0.20.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path
    from sympy import cos, sin, solve

    import mechanics.space as space
    from mechanics import Markdown, diff, base_space, constants, group, index, variable, variables
    from mechanics.differential_equation import to_first_order
    from mechanics.integrator.runge_kutta import rk4_explicit
    from mechanics.lagrange import euler_lagrange_equation
    from mechanics.solver import build_solver


@app.cell
def intro():
    mo.md("""
    # Kepler problem
    """)
    return


@app.cell
def figure():
    mo.image(src=Path(__file__).with_name("figure") / "kepler_orbit.svg")
    return


@app.cell
def derive_lagrangian():
    t = base_space("t")

    q = group(
        variable("r", t),
        variable(r"\theta", t, space=space.S)
    )

    c = constants(r"\mu m")

    f = group(
        x = q.r * cos(q.theta),
        y = q.r * sin(q.theta),
        U = -c.mu * c.m / q.r,
        K = lambda f: (c.m / 2 * (diff(f.x, t) ** 2 + diff(f.y, t) ** 2)).simplify(),
        E = lambda f: f.K + f.U
    )
    L = f.K - f.U
    return L, c, f, q, t


@app.cell
def derive_equations(L, q, t):
    EL = euler_lagrange_equation(L, q)
    F = solve(EL, diff(q, t, 2))
    return EL, F


@app.cell(hide_code=True)
def show_symbolic_results(EL, F, L):
    md = Markdown()
    md.add_markdown("### Lagrangian")
    md.show("L = ", L)
    md.add_markdown("### Euler-Lagrange equations")
    md.show_equations(EL)
    md.add_markdown("### Solved equations of motion")
    md.show_equations(F)
    md.render(mo)
    return


@app.cell
def build_step_equations(F, q):
    h, T = constants("h T")
    i = index("i")

    first_order = to_first_order(F)
    rk4 = rk4_explicit(first_order.equations, h, i)
    d = rk4.conversion * first_order.conversion

    q_ = d(q)
    v_ = d(first_order.variables_of_order(1))
    return T, d, h, i, q_, rk4, v_


@app.cell
def build_solver_module(T, c, d, f, h, i, rk4):
    solver = build_solver()
    solver.constants(*c)
    solver.constants(h, T)
    solver.variables(*rk4.variables, index=(i, 0, T / h))
    solver.functions(*f._fields, index=(i, 0, T / h))
    solver.inputs(*(x_[0] for x_ in rk4.state_variables))
    with solver.steps(i, 0, T / h) as step:
        step.explicit(rk4.step_equations)
        step.calculate(d(f), i)
    solver = solver.generate()
    return (solver,)


@app.cell
def controls():
    sliders = mo.ui.dictionary(
        {
            "r_0": mo.ui.slider(start=0.1, stop=2.0, step=0.01, value=1.0, label=r"$r(0)$"),
            "theta_0": mo.ui.slider(start=-3.14, stop=3.14, step=0.01, value=0.0, label=r"$\theta(0)$"),
            "v_r_0": mo.ui.slider(start=-1.0, stop=1.0, step=0.01, value=0.0, label=r"$\dot{r}(0)$"),
            "v_theta_0": mo.ui.slider(start=0.1, stop=2.0, step=0.01, value=1.1, label=r"$\dot{\theta}(0)$"),
            "h": mo.ui.number(start=0.001, stop=0.1, step=0.001, value=0.01, label=r"$h$"),
            "T": mo.ui.number(start=1.0, stop=5000.0, step=1.0, value=50.0, label=r"$T$"),
        }
    )
    sliders
    return (sliders,)


@app.cell
def run_simulation(T, c, h, q_, sliders, solver, v_):
    sv = sliders.value
    result = solver.run(
        {
            c.mu: 1.0,
            c.m: 1.0,
            q_.r[0]: sv["r_0"],
            q_.theta[0]: sv["theta_0"],
            v_.r[0]: sv["v_r_0"],
            v_.theta[0]: sv["v_theta_0"],
            h: sv["h"],
            T: sv["T"],
        }
    )
    t_ = np.arange(0, result[T], result[h])
    return result, t_


@app.cell
def plot_trajectory(result):
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "cm"

    _fig, _ax = plt.subplots(1, 1, figsize=(4, 4), tight_layout=True)
    _ax.plot(result["x"], result["y"])
    _ax.set_xlabel("$x$")
    _ax.set_ylabel("$y$")
    _ax.set_aspect("equal")
    _fig
    return


@app.cell
def plot_timeseries(q_, result, t_, v_):
    _fig2, axes = plt.subplots(3, 2, figsize=(8, 8), tight_layout=True)

    series = [
        (r"r", q_.r),
        (r"\theta", q_.theta),
        (r"\dot{r}", v_.r),
        (r"\dot{\theta}", v_.theta),
        ("E", "E"),
    ]
    for ax, (name, var) in zip(axes.flatten(), series):
        n = min(len(t_), len(result[var]))
        ax.plot(t_[:n], result[var][:n])
        ax.set_xlabel("$t$")
        ax.set_ylabel(f"${name}$")
        ax.set_xlim(0, t_[-1])
    _fig2
    return


if __name__ == "__main__":
    app.run()
