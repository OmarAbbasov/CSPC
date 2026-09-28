# CSPC - Computer Science for Physics and Chemistry

My coursework repository. Each practical is under PW<n>/Lab <X>/.

## Setup
Create the environment for a given lab:
    conda env create -f PW<n>/Lab\ <X>/environment.yml
    conda activate cspc

## PW1 - Lab A: Reproducible Foundations
**What I built:**

Created the CSPC repository structure, configured conda environment, integrated Git/GitHub workflow, written unit tests with pytest, and benchmarked NumPy against Python loops.

**Speed comparison (loop vs NumPy):**

- loop : 3.0120 s
- numpy : 0.0004 s
- speed-up: 8466.98x faster
**Tests:** all passing? yes

**Conclusion:**

NumPy vectorization dramatically increases computational performance compared to pure Python loops. Setting up isolated environments and Git repositories guarantees complete reproducibility of scientific results.

## PW1 Lab B Report

* **Data Observation:** The observed decay count matches the theoretical exponential decay law ($N_0 e^{-\lambda t}$) closely with $\lambda=0.3$.
* **Snakemake Pipeline:** The Snakemake pipeline automates the generation of `figure.png` from `decay_observed.csv` and `plot.py`, ensuring the figure is only rebuilt when input data or scripts change.

## PW2 - Lab A: Motion from Tracking Data

**What I built:**
Analyzed noisy free-fall position tracking data. Calculated velocity and acceleration using numerical differentiation (`np.gradient`), then integrated the noisy acceleration back to recover position using `scipy.integrate.cumulative_trapezoid`.

**Results:**
* Mean Acceleration: ~ -9.81 m/s² (close to theoretical -g)
* Acceleration Noise: High standard deviation in acceleration compared to position
* Max Recovered Position Error: < 1.0 m

**Why Acceleration is Noisy:**
Numerical differentiation compares adjacent noisy data points and divides by a small time step ($\Delta t$), which amplifies high-frequency measurement noise with every derivative step; whereas integration accumulates and sums values, causing random noise to cancel out.