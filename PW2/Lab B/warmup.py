"""
PW2 Lab B Part 2 -- three routes to a minimum.

Compare gradient descent, Newton, and SLSQP on two functions:
  2A: f(x) = (x-3)**2 + 1          (easy, one minimum at x=3)
  2B: g(x) = x**4 - 3*x**2 + x + 5 (harder, several stationary points)
Run:  python warmup.py
"""
import numpy as np
from scipy.optimize import newton, minimize

# ---------- 2A: easy convex function ----------
def f(x):   return (x-3)**2 + 1
def df(x):  return 2*(x-3)
def d2f(x): return 2.0

print("=== 2A: Easy Convex Function ===")

# (1) Gradient Descent by hand
x_gd = 0.0
lr = 0.1
for _ in range(1000):
    grad = df(x_gd)
    if abs(grad) < 1e-6:
        break
    x_gd -= lr * grad
print(f"Gradient Descent: x = {x_gd:.4f}")

# (2) Newton's method on df
x_newton = newton(df, 0.0, fprime=d2f)
print(f"Newton method:    x = {x_newton:.4f}")

# (3) SLSQP
res_slsqp = minimize(f, 0.0, method="SLSQP")
print(f"SLSQP method:     x = {res_slsqp.x[0]:.4f}\n")


# ---------- 2B: harder landscape ----------
def g(x):   return x**4 - 3*x**2 + x + 5
def dg(x):  return 4*x**3 - 6*x + 1
def d2g(x): return 12*x**2 - 6

print("=== 2B: Harder Landscape ===")

for x0 in [0.0, 2.0]:
    print(f"\n--- Starting from x0 = {x0} ---")
    
    # GD
    x_gd = x0
    lr = 0.01
    for _ in range(1000):
        x_gd -= lr * dg(x_gd)
    print(f"GD:     x = {x_gd:.4f}, g(x) = {g(x_gd):.4f}")
    
    # Newton
    x_newt = newton(dg, x0, fprime=d2g)
    d2_val = d2g(x_newt)
    kind = "minimum" if d2_val > 0 else ("maximum" if d2_val < 0 else "saddle")
    print(f"Newton: x = {x_newt:.4f}, g''(x) = {d2_val:.4f} ({kind})")
    
    # SLSQP
    res = minimize(g, x0, method="SLSQP")
    print(f"SLSQP:  x = {res.x[0]:.4f}, g(x) = {g(res.x[0]):.4f}")