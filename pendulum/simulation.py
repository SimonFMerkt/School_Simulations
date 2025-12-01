# simulate.py

import numpy as np
from scipy.integrate import solve_ivp
import params
import os


def pendulum_ode(t, y, i) -> list[float]:
    theta, omega = y
    dtheta_dt = omega
    domega_dt = -(params.g / params.L) * np.sin(theta) - params.b[i] * omega
    return [dtheta_dt, domega_dt]

def compute_cinetic_energy(omega) -> float:
    return 0.5 * (params.L ** 2) * (omega ** 2)

def compute_potential_energy(theta) -> float:
    return params.g * params.L * (1 - np.cos(theta))   

def run_simulation(i):
    t_eval = np.linspace(params.t_start, params.t_end, params.num_steps)
    y0 = [params.theta0[i], params.omega0[i]]

    sol = solve_ivp(
        pendulum_ode,
        (params.t_start, params.t_end),
        y0,
        t_eval=t_eval,
        method="RK45",
        args=(i,)
    )
    if not os.path.exists(params.output_folder):
        os.makedirs(params.output_folder)

    np.savez(
        params.output_folder + '/' + params.output_file[i],
        t=sol.t,
        theta=sol.y[0],
        omega=sol.y[1]
    )

    print(f"Saved simulation results to {params.output_file[i]}")


if __name__ == "__main__":
    for i in range(len(params.theta0)):
        run_simulation(i)
