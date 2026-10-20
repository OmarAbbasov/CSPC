"""
PW2 Lab B Part 5 (bonus) -- find a titration's equivalence point.

titration.csv holds a titration curve: pH versus the volume of base added.
The equivalence point is the volume where the pH changes fastest (the steep
jump). Numerically, that is where the SLOPE of the pH curve is largest.
Run:  python titration.py
"""
import numpy as np
import matplotlib.pyplot as plt

# TODO 1: Read titration.csv
data = np.loadtxt("titration.csv", delimiter=",", skiprows=1)
V = data[:, 0]
pH = data[:, 1]

# TODO 2: Slope and equivalence point
slope = np.gradient(pH, V)
eq_index = np.argmax(slope)
v_eq = V[eq_index]

print(f"Equivalence Point Volume: {v_eq:.2f} mL")

# TODO 3: Plots side-by-side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Left plot
ax1.plot(V, pH, color="blue", label="pH Curve")
ax1.axvline(v_eq, color="red", linestyle="--", label=f"Eq Point ({v_eq:.2f} mL)")
ax1.set_xlabel("Volume of Base (mL)")
ax1.set_ylabel("pH")
ax1.set_title("Titration Curve")
ax1.grid(True)
ax1.legend()

# Right plot
ax2.plot(V, slope, color="green", label="dpH/dV")
ax2.axvline(v_eq, color="red", linestyle="--", label=f"Peak Slope ({v_eq:.2f} mL)")
ax2.set_xlabel("Volume of Base (mL)")
ax2.set_ylabel("Slope (dpH/dV)")
ax2.set_title("First Derivative of pH")
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.savefig("titration.png")
print("Saved titration.png")