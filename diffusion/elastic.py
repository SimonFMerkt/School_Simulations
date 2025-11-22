import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def particle_collision(p1, v1, p2, v2, m1=1, m2=1):
    """
    Resolve 2D elastic collision between two circles.
    """
    deltax = p2 - p1
    deltav = v1 - v2
    dist = np.linalg.norm(deltax)
    
    mass1 = 2*m2/(m1+m2)
    mass2 = 2*m1/(m1+m2)

    with np.errstate(divide='ignore', invalid='ignore'):
        v1_new = v1 + mass1 * np.linalg.vecdot(deltav, -deltax) / (dist**2) * deltax
        v2_new = v2 - mass2 * np.linalg.vecdot(-deltav, deltax) / (dist**2) * deltax
  

    return v1_new, v2_new


def wall_collision(p, v, r, box_min=0.0, box_max=1.0):
    for i in range(2):
        if p[i] - r < box_min:
            p[i] = box_min + r
            v[i] *= -1
        elif p[i] + r > box_max:
            p[i] = box_max - r
            v[i] *= -1
    return p, v

def center_wall_collision(p, v, r, center=0.5):
    if abs(p[0] - center) < r:
        if p[0] < center:
            p[0] = center - r
        else:
            p[0] = center + r
        v[0] *= -1
    return p, v