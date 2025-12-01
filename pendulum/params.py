# params.py

import numpy as np

# Physical parameters
g: float = 9.81          # gravity (m/s^2)
L: float = 1.0           # length of pendulum (m)
b: list[float] = [0, 0.5]          # damping coefficient (kg/s). Set to 0 for no damping.

# Initial conditions
theta0: list[float] = [np.deg2rad(60), np.deg2rad(60)]   # initial angle in radians
omega0: list[float] = [0, 0]              # initial angular velocity

# Simulation time parameters
t_start: float = 0.0
t_end: float = 10.0
num_steps: int = 4000

# Output file
output_folder: str = "simulations"
output_file: list[str] = ["undamped_60_0.npz", "damped_60_0.npz"]