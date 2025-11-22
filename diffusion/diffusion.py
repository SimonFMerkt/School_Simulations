import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.lines import Line2D
import yaml

from matplotlib.widgets import Button
from elastic import particle_collision, wall_collision, center_wall_collision


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
})

# -----------------------------------------
# Simulation Parameters
# -----------------------------------------
def load_config(path="diffusion_parameters.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)
config = load_config()

num_partA, num_partB = config["particle_numbers"].values()
radiusA, radiusB = config["particle_sizes"].values() 
massA, massB = config["particle_masses"].values()
box_min, box_max, box_size = config["box_dimensions"].values()
TA, TB = config["gas_temperatures"].values()
dt, steps = config["simulation_parameters"].values() 
wall_exists = True

num_particles = num_partA + num_partB
radii = 0.01*np.concatenate((radiusA*np.ones(num_partA), radiusB*np.ones(num_partB)))
masses = np.concatenate((massA*np.ones(num_partA), massB*np.ones(num_partB)))

# -----------------------------------------
# Initialization
# -----------------------------------------

# Random initial positions

positions = np.zeros((num_particles, 2))
if num_partB != 0:
    # Gas A on left half
    positions[:num_partA, 0] = np.random.rand(num_partA) * 0.45*box_max + 0.01*radiusA
    positions[:num_partA, 1] = np.random.rand(num_partA) * (box_size - 2*0.01*radiusA) + 0.01*radiusA

    # Gas B on right half
    positions[num_partA:, 0] = np.random.rand(num_partB) * 0.45*box_max + 0.55*box_max
    positions[num_partA:, 1] = np.random.rand(num_partB) * (box_size - 2*0.01*radiusB) + 0.01*radiusB
else:
    positions = np.random.rand(num_particles,2)

# Velocities scaled to match temperature
velocities = np.random.randn(num_particles, 2)
velocities[:num_partA] *= np.sqrt(TA) / np.sqrt(np.mean(np.sum(velocities[:num_partA]**2, axis=1)))
velocities[num_partA:] *= np.sqrt(TB) / np.sqrt(np.mean(np.sum(velocities[num_partA:]**2, axis=1)))


# -----------------------------------------
# Figure Setup
# -----------------------------------------

fig, ax_sim = plt.subplots()

# Gas simulation panel
ax_sim.set_xlim(0, box_max)
ax_sim.set_ylim(0, box_max)
ax_sim.set_aspect("equal")
ax_sim.set_title("Diffusion Simulation")
ax_sim.get_xaxis().set_visible(False)
ax_sim.get_yaxis().set_visible(False)


legend_elements = [
    Line2D([0], [0], marker='o', color='w',
           markerfacecolor='blue', markersize=10,
           label=f"Gas A: {num_partA} particles at $T_A = {TA}$"),
    Line2D([0], [0], marker='o', color='w',
           markerfacecolor='red', markersize=10,
           label=f"Gas B: {num_partB} particles at $T_B = {TB}$")
]

ax_sim.legend(handles=legend_elements, loc='upper right')

(center_line,) = ax_sim.plot([box_max/2, box_max/2], [0,box_max], color='black', lw=3)
# Create circles
c = []
for i in range(num_partA):
    c.append(plt.Circle(positions[i], radii[i], color='blue'))
    ax_sim.add_patch(c[i])
for i in range(num_partB):
    c.append(plt.Circle(positions[num_partA+i], radii[num_partA+i], color='red'))
    ax_sim.add_patch(c[num_partA+i])

# --------------------------
# Button to remove wall
# --------------------------
def remove_wall(event):
    global wall_exists
    wall_exists = False
    center_line.set_visible(False)
    fig.canvas.draw_idle()

button_ax = fig.add_axes([0.4, 0.03, 0.2, 0.08])
button = Button(button_ax, "Remove Dividor", color='lightgray', hovercolor='gray')
button.on_clicked(remove_wall)

# -----------------------------------------
# Update Function
# -----------------------------------------

def update(frame):
    global positions, velocities

    # Collisions
    for i in range(num_particles):
        positions[i], velocities[i] = wall_collision(positions[i], velocities[i], radii[i], box_min, box_max)
        if wall_exists:
            positions[i], velocities[i] = center_wall_collision(positions[i], velocities[i], radii[i], box_max/2)

    for j in range(num_particles):
        for i in range(j+1, num_particles):

                dist = np.linalg.norm(positions[i] - positions[j])
                if dist < radii[i] + radii[j]:
  
                    velocities[i], velocities[j] = particle_collision(positions[i], velocities[i], positions[j], velocities[j], masses[i], masses[j])
                    with np.errstate(divide='ignore', invalid='ignore'):
                        overlap = (radii[i] + radii[j]) - dist
                        direction = (positions[j] - positions[i]) / dist
                        positions[i] -= direction * overlap * 0.5
                        positions[j] += direction * overlap * 0.5
    

    # Move particles
    positions += velocities * dt
   
    for i in range(num_particles):
        c[i].center = positions[i]
    return c

# -----------------------------------------
# Run Animation
# -----------------------------------------

anim = animation.FuncAnimation(fig, update, frames = steps, interval=10)
plt.show()
