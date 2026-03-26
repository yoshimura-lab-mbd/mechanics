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
    from mechanics import Markdown, base_space, constants, index, variable, variables
    from mechanics.differential_equation import to_first_order
    from mechanics.integrator.euler import (
        backward_euler_explicit,
        euler_explicit,
        modified_euler_explicit,
    )
    from mechanics.integrator.runge_kutta import rk4_explicit
    from mechanics.lagrange import euler_lagrange_equation
    from mechanics.solver import build_solver

    return (
        Markdown,
        backward_euler_explicit,
        base_space,
        build_solver,
        constants,
        cos,
        diff,
        euler_explicit,
        euler_lagrange_equation,
        index,
        mo,
        modified_euler_explicit,
        np,
        plt,
        rk4_explicit,
        sin,
        solve,
        space,
        to_first_order,
        variable,
        variables,
    )


@app.cell
def intro(mo):
    mo.md("""
    # Kepler problem: Integrator comparison

    Compare different numerical integrators for the Kepler problem.
    """)
    return


@app.cell
def derive_lagrangian(base_space, constants, cos, diff, sin, space, variable):
    t = base_space("t")

    def dot(f):
        return diff(f, t)

    r_var = variable("r", t)
    theta_var = variable(r"\theta", t, space=space.S)
    q = (r_var, theta_var)

    dq = tuple(dot(q_n) for q_n in q)
    ddq = tuple(dot(dq_n) for dq_n in dq)

    c = constants(r"\mu m")

    x = r_var * cos(theta_var)
    y = r_var * sin(theta_var)

    U = -c.mu / r_var
    kinetic = (c.m / 2 * (dot(x) ** 2 + dot(y) ** 2)).simplify()
    E = kinetic + U
    L = kinetic - U
    return E, L, c, ddq, dq, q, x, y


@app.cell
def derive_equations(L, ddq, euler_lagrange_equation, q, solve):
    EL = euler_lagrange_equation(L, q)
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
def setup_constants(constants, index):
    ht = constants("h T")
    h = ht.h
    T = ht.T
    i = index("i")

    return h, i, T


@app.cell
def build_integrators(
    F,
    backward_euler_explicit,
    dq,
    euler_explicit,
    h,
    i,
    modified_euler_explicit,
    q,
    rk4_explicit,
    to_first_order,
):
    # Convert to first-order form
    first_order = to_first_order(F)

    integrators = {}

    # Euler explicit
    euler = euler_explicit(first_order.equations, h, i)
    integrators["Euler"] = {
        "result": euler,
        "states": euler.state_variables,
        "stages": (),
        "step_eq": euler.step_equations,
        "conversion": euler.conversion,
    }

    # Backward Euler
    b_euler = backward_euler_explicit(first_order.equations, h, i)
    integrators["Backward Euler"] = {
        "result": b_euler,
        "states": b_euler.state_variables,
        "unknowns": b_euler.unknowns,
        "stages": (),
        "step_eq": b_euler.step_equations,
        "conversion": b_euler.conversion,
    }

    # Heun (modified Euler)
    heun = modified_euler_explicit(first_order.equations, h, i)
    integrators["Heun"] = {
        "result": heun,
        "states": heun.state_variables,
        "stages": heun.stage_variables,
        "step_eq": heun.step_equations,
        "conversion": heun.conversion,
    }

    # RK4
    rk4 = rk4_explicit(first_order.equations, h, i)
    integrators["RK4"] = {
        "result": rk4,
        "states": rk4.state_variables,
        "stages": rk4.stage_variables,
        "step_eq": rk4.step_equations,
        "conversion": rk4.conversion,
    }

    # Compute converted variables using first_order + rk4 conversion
    d = rk4.conversion * first_order.conversion
    r_, theta_ = d(q)
    v_r_, v_theta_ = d(dq)

    return first_order, integrators, r_, theta_, v_r_, v_theta_


@app.cell
def build_solvers(E, T, build_solver, c, first_order, h, i, integrators, x, y):
    solvers = {}

    for method, integ in integrators.items():
        solver = build_solver()
        solver.constants(*c)
        solver.constants(h, T)

        states = integ["states"]
        stages = integ["stages"]
        solver.variables(*states, *stages, index=(i, 0, T / h))
        solver.functions("x", "y", "E", index=(i, 0, T / h))
        solver.inputs(*(x_[0] for x_ in states))

        # Combined conversion for this integrator
        d = integ["conversion"] * first_order.conversion

        with solver.steps(i, 0, T / h) as step:
            if method == "Backward Euler":
                # Backward Euler needs Newton solver
                with step.newton() as newton:
                    newton.unknowns(*integ["unknowns"])
                    newton.initial_guess({v: v.subs(i, i - 1) for v in integ["unknowns"]})
                    newton.implicit(integ["step_eq"])
            else:
                # Explicit methods
                step.explicit(integ["step_eq"])
            step.calculate({"x": d(x), "y": d(y), "E": d(E)}, i)

        solvers[method] = solver.generate()

    return solvers


@app.cell
def controls(mo):
    sliders = mo.ui.dictionary(
        {
            "mu": mo.ui.slider(start=0.1, stop=10.0, step=0.1, value=1.0, label=r"$\mu$"),
            "m": mo.ui.slider(start=0.1, stop=10.0, step=0.1, value=1.0, label=r"$m$"),
            "r_0": mo.ui.slider(start=0.1, stop=5.0, step=0.1, value=1.0, label=r"$r(0)$"),
            "theta_0": mo.ui.slider(start=-3.14, stop=3.14, step=0.01, value=0.0, label=r"$\theta(0)$"),
            "v_r_0": mo.ui.slider(start=-5.0, stop=5.0, step=0.1, value=0.0, label=r"$\dot{r}(0)$"),
            "v_theta_0": mo.ui.slider(start=0.1, stop=5.0, step=0.1, value=1.1, label=r"$\dot{\theta}(0)$"),
            "h": mo.ui.slider(start=0.001, stop=0.5, step=0.01, value=0.2, label=r"$h$"),
            "T": mo.ui.slider(start=10.0, stop=200.0, step=10.0, value=100.0, label=r"$T$"),
        }
    )
    sliders
    return (sliders,)


@app.cell
def run_simulations(T, c, h, r_, sliders, solvers, theta_, v_r_, v_theta_):
    sv = sliders.value
    results = {}

    for method, solver in solvers.items():
        results[method] = solver.run(
            {
                c.mu: sv["mu"],
                c.m: sv["m"],
                r_[0]: sv["r_0"],
                theta_[0]: sv["theta_0"],
                v_r_[0]: sv["v_r_0"],
                v_theta_[0]: sv["v_theta_0"],
                h: sv["h"],
                T: sv["T"],
            }
        )

    return results


@app.cell
def plot_orbits(np, plt, r_, results, theta_):
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "cm"

    _fig, _ax = plt.subplots(1, 1, figsize=(6, 6), tight_layout=True)
    _ax.set_xlabel("$x$")
    _ax.set_ylabel("$y$")
    _ax.set_aspect("equal")
    _ax.set_xlim(-2, 2)
    _ax.set_ylim(-2, 2)

    for method, result in results.items():
        _ax.plot(result["x"], result["y"], label=method, linewidth=2)

    _ax.legend()
    _ax.set_title("Orbital comparison")
    _fig
    return


@app.cell
def plot_timeseries(np, plt, r_, results, sliders, theta_, v_r_, v_theta_):
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "cm"

    sv = sliders.value
    h_val = sv["h"]

    _fig, axes = plt.subplots(3, 2, figsize=(10, 10), tight_layout=True)
    axes = axes.flatten()

    series = [
        (r"r", r_),
        (r"\theta", theta_),
        (r"\dot{r}", v_r_),
        (r"\dot{\theta}", v_theta_),
        ("E", "E"),
    ]

    for idx, (name, var) in enumerate(series):
        ax = axes[idx]
        ax.set_xlabel("$t$")
        ax.set_ylabel(f"${name}$")

        # Set y-axis limits based on RK4 result
        rk4_result = results["RK4"][var][:-1]
        if hasattr(rk4_result, 'max'):
            value_range = rk4_result.max() - rk4_result.min()
            value_center = (rk4_result.max() + rk4_result.min()) / 2
            ax.set_ylim(value_center - value_range * 0.6, value_center + value_range * 0.6)

        for method, result in results.items():
            n_points = len(result[var][:-1])
            t_data = np.arange(n_points) * h_val
            ax.plot(t_data, result[var][:-1], label=method, linewidth=1.5)

        ax.legend()
        ax.grid(True, alpha=0.3)

    _fig
    return


if __name__ == "__main__":
    app.run()
