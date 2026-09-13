import time
from decay import simulate, simulate_loop

N0 = 200000
lam = 0.4

t0 = time.perf_counter()
res_loop = simulate_loop(N0, lam)
t1 = time.perf_counter()
time_loop = t1 - t0

t0 = time.perf_counter()
res_numpy = simulate(N0, lam)
t1 = time.perf_counter()
time_numpy = t1 - t0

speedup = time_loop / time_numpy

print(f"Pure Python loop time: {time_loop:.4f} s")
print(f"NumPy vectorized time: {time_numpy:.4f} s")
print(f"NumPy is {speedup:.2f}x faster!")