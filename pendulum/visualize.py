# visualize.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import params
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import simulation

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
})

def visualize():

    fig = plt.figure(figsize=(12, 6))

    # Pendulum animation (left)
    ax1 = fig.add_subplot(121)
    ax1.set_xlim(-params.L * 1.2, params.L * 1.2)
    ax1.set_ylim(-params.L * 1.2, params.L * 1.2)
    ax1.set_title("$\mathrm{Pendulum} \\ \mathrm{Animation}$")
    ax1.set_xlabel("$x$")
    ax1.set_ylabel("$y$")
    sub_ax = inset_axes(ax1, width="80%", height="30%", loc='upper center', borderpad=2.5)
    sub_ax.set_title('$\mathrm{Energy}$')
    sub_ax.set_xlabel('$t$')
    sub_ax.set_ylabel('$E$')

    # Phase space (right)
    ax2 = fig.add_subplot(122)
    ax2.set_title("$\mathrm{Phase} \\ \mathrm{Space}$")
    ax2.set_xlabel("$\\theta$")
    ax2.set_ylabel("$\\omega$")
    ax2.set_xticks(np.linspace(-np.pi/4, np.pi/4, 3))
    ax2.set_xticklabels(["$-\\frac{\\pi}{4}$", "0", "$\\frac{\\pi}{4}$"])    


    thetas = []
    omegas = []
    xs = []
    ys = []
    times = None  # assume all files have the same time vector
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
    for filename in params.output_file:
        data = np.load(params.output_folder + '/' + filename)

        t = data["t"]
        theta = data["theta"]
        theta_centered = (theta + np.pi) % (2 * np.pi) - np.pi  #
        
        theta = theta_centered
        
        omega = data["omega"]
        
        if times is None:
            times = t  # Save time vector from first file

        # store for animation
        thetas.append(theta)
        omegas.append(omega)
        xs.append(params.L * np.sin(theta))
        ys.append(-params.L * np.cos(theta))

        # draw full trajectory in phase space (background)
        ax2.plot(theta, omega, color=colors[len(thetas)-1], alpha=0.3)
        #sub_ax.plot(t, simulation.compute_cinetic_energy(omega), ':',color=colors[len(thetas)-1], label=filename)
        #sub_ax.plot(t, simulation.compute_potential_energy(theta), '--', color=colors[len(thetas)-1])
        sub_ax.plot(t, simulation.compute_cinetic_energy(omega) + simulation.compute_potential_energy(theta), color=colors[len(thetas)-1])

    n_pendulums = len(thetas)

    lines = []        # pendulum rods
    bobs = []         # pendulum masses
    phase_points = [] # red points in phase space

    for i in range(n_pendulums):
        color = colors[i % len(colors)]
        line, = ax1.plot([], [], lw=2, color=color)
        bob,  = ax1.plot([], [], 'o', color = color, markersize=10)
        phase, = ax2.plot([], [], 'o', color=color)

        lines.append(line)
        bobs.append(bob)
        phase_points.append(phase)

    # --------------------
    #  ANIMATION FUNCTION
    # --------------------
    def update(frame):

        for i in range(n_pendulums):
            # Pendulum
            lines[i].set_data([0, xs[i][frame]], [0, ys[i][frame]])
            bobs[i].set_data([xs[i][frame]], [ys[i][frame]])

            # Phase space
            phase_points[i].set_data([thetas[i][frame]], [omegas[i][frame]])

        return (*lines, *bobs, *phase_points)

    # -----------------
    #  CREATE ANIMATION
    # -----------------
    #interval_ms = (times[1] - times[0])/2000

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(times),
        interval=10,
        blit=True
    )


    #plt.tight_layout()
    plt.show()
    ani.save("pendulum.mp4", writer="ffmpeg", dpi=200)


if __name__ == "__main__":
    visualize()
