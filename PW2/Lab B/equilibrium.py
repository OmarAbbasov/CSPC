"""
PW2 Lab B Part 4 -- chemical equilibrium via the equilibrium constant K.

Reaction  H2 + I2 <=> 2 HI, starting from 1 mol H2 and 1 mol I2.
As the reaction proceeds by an extent x:  H2 = 1-x,  I2 = 1-x,  HI = 2x.
At equilibrium the composition satisfies the equilibrium constant
        K = [HI]^2 / ([H2][I2]) = (2x)^2 / ((1-x)(1-x)).
Given K, find the extent x. Solve it TWO ways and compare.
Run:  python equilibrium.py
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton, minimize

K = 15.6
a = b = 1.0

# TODO 1: Imbalance function
def k_imbalance(x):
    return ((2*x)**2) / ((a - x)*(b - x)) - K

# TODO 2: Newton's method (root-finding)
x_newton = newton(k_imbalance, x0=0.5)

# TODO 3: SLSQP minimization of squared imbalance
def square_imbalance(x):
    return k_imbalance(x[0])**2

res = minimize(square_imbalance, x0=[0.5], method="SLSQP", bounds=[(0, 0.999)])
x_slsqp = res.x[0]

print(f"Newton extent x: {x_newton:.4f}")
print(f"SLSQP extent x:  {x_slsqp:.4f}")

# TODO 4: Equilibrium amounts and Plot
H2_eq = a - x_newton
I2_eq = b - x_newton
HI_eq = 2 * x_newton

print(f"Equilibrium amounts: H2 = {H2_eq:.4f} mol, I2 = {I2_eq:.4f} mol, HI = {HI_eq:.4f} mol")

x_vals = np.linspace(0, 0.99, 100)
plt.figure(figsize=(7, 5))
plt.plot(x_vals, a - x_vals, label="H2")
plt.plot(x_vals, b - x_vals, label="I2")
plt.plot(x_vals, 2 * x_vals, label="HI")
plt.axvline(x_newton, color="black", linestyle="--", label=f"Equilibrium (x={x_newton:.2f})")
plt.xlabel("Extent of reaction (x)")
plt.ylabel("Amount (mol)")
plt.title("Chemical Equilibrium: H2 + I2 <=> 2 HI")
plt.grid(True)
plt.legend()
plt.savefig("equilibrium.png")
print("Saved equilibrium.png")