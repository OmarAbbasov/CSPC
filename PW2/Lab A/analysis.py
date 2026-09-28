"""
PW2 Lab A -- Motion from tracking data.

Read noisy free-fall position measurements, then:
  - differentiate once  -> velocity
  - differentiate twice -> acceleration
  - integrate the acceleration back up -> recover velocity and position

Complete the TODOs. Run: python analysis.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid

# --- TODO 1: Read freefall.csv into arrays t and y ---
# Measurements of height y sampled every 0.1s
data = np.loadtxt("freefall.csv", delimiter=",", skiprows=1)
t = data[:, 0]
y = data[:, 1]

# --- TODO 2: Compute velocity and acceleration via numerical differentiation ---
v = np.gradient(y, t)  # First derivative: v = dy/dt
a = np.gradient(v, t)  # Second derivative: a = dv/dt

mean_a = np.mean(a)
std_a = np.std(a)

print(f"Mean Acceleration: {mean_a:.4f} m/s^2")
print(f"Standard Deviation of Acceleration: {std_a:.4f} m/s^2")

# --- TODO 3: Integrate acceleration back up to recover velocity and position ---
# Integration: cumulative_trapezoid adds up values step-by-step (+ initial value)
rec_v = cumulative_trapezoid(a, t, initial=0) + v[0]
rec_y = cumulative_trapezoid(rec_v, t, initial=0) + y[0]

max_diff = np.max(np.abs(y - rec_y))
print(f"Max difference between original and recovered position: {max_diff:.4f} m")

# --- TODO 4: Make a figure with 3 stacked panels ---
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

# Panel 1: Position
ax1.plot(t, y, label="Measured Position", color="blue")
ax1.set_ylabel("Position (m)")
ax1.set_title("Free-Fall Motion Analysis")
ax1.grid(True)
ax1.legend()

# Panel 2: Velocity
ax2.plot(t, v, label="Velocity (dy/dt)", color="orange")
ax2.set_ylabel("Velocity (m/s)")
ax2.grid(True)
ax2.legend()

# Panel 3: Acceleration
ax3.plot(t, a, label="Acceleration (dv/dt)", color="red", alpha=0.6)
ax3.axhline(-9.81, color="black", linestyle="--", label="True -9.81 m/s²")
ax3.set_xlabel("Time (s)")
ax3.set_ylabel("Acceleration (m/s²)")
ax3.grid(True)
ax3.legend()

plt.tight_layout()
plt.savefig("motion.png")
print("Saved figure as motion.png")