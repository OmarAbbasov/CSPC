"""
PW2 Lab B Part 3 -- fit a reaction's rate constant to measured data.

A first-order reaction decays as  C(t) = C0 * exp(-k*t).  You have noisy
concentration-vs-time measurements; find the k that best matches them.
Complete the TODOs. Run:  python kinetics.py
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# TODO 1: Read kinetics.csv
data = np.loadtxt("kinetics.csv", delimiter=",", skiprows=1)
t = data[:, 0]
C = data[:, 1]
C0 = C[0]

# TODO 2: Total sum of squared errors
def total_error(k):
    model = C0 * np.exp(-k * t)
    return np.sum((C - model)**2)

# TODO 3: Minimise total_error using SLSQP
res = minimize(total_error, x0=0.5, method="SLSQP", bounds=[(0, 5)])
k_fitted = res.x[0]
print(f"Fitted rate constant k = {k_fitted:.4f}")

# TODO 4: Plot
plt.figure(figsize=(7, 5))
plt.scatter(t, C, color="red", label="Measured Data")
t_dense = np.linspace(min(t), max(t), 100)
plt.plot(t_dense, C0 * np.exp(-k_fitted * t_dense), color="blue", label=f"Fitted Model (k={k_fitted:.4f})")
plt.xlabel("Time")
plt.ylabel("Concentration")
plt.title("Reaction Kinetics Fit")
plt.grid(True)
plt.legend()
plt.savefig("kinetics.png")
print("Saved kinetics.png")