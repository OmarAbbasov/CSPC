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

## PW2 - Lab B: Optimization in Chemistry

**1. Optimization Methods Comparison (Part 2):**
* **Convex function $f(x)$:** All three methods (Gradient Descent, Newton, SLSQP) successfully converged to the global minimum at $x \approx 3.00$.
* **Complex function $g(x)$:** 
  * Starting at $x_0 = 0$: Newton's method converged to a local maximum/stationary point ($g''(x) < 0$), whereas GD and SLSQP successfully located local/global minima.
  * Starting at $x_0 = 2$: All methods converged towards the global minimum near $x \approx 1.16$.
  * *Takeaway:* Method choice and initial guesses $x_0$ strongly affect convergence on complex non-convex landscapes.

**2. Chemistry Applications:**
* **Reaction Rate Fit:** Fitted rate constant $k \approx 0.25$, accurately capturing the first-order exponential decay curve.
* **Chemical Equilibrium ($H_2 + I_2 \rightleftharpoons 2HI$):** Both Newton (root-finding) and SLSQP (minimization) agreed on the reaction extent $x \approx 0.66$, corresponding to $HI \approx 1.33 \text{ mol}$ and $H_2 = I_2 \approx 0.34 \text{ mol}$.
* **Titration Equivalence Point:** Maximum slope of $dpH/dV$ identified the equivalence point at $V \approx 50.00 \text{ mL}$.

## PW3 - Data, Distributions, Testing, and Causation

### Session 1: Characterize the Data
* **Distributions:**
  * `age`: Symmetric, nearly bell-shaped distribution.
  * `thalach`: Approximately normal with a slight left skew.
  * `chol` & `trestbps`: Right-skewed continuous variables due to extreme upper medical outliers.
* **Normality Verdicts:**
  * Shapiro-Wilk statistical tests confirm that `thalach` and `age` closely approximate normality ($p > 0.05$ or near-threshold), whereas `chol` and `trestbps` deviate significantly due to skewness ($p < 0.05$).

### Session 2: Draw Conclusions
* **Heart Disease Effect:** 
  * Patients with heart disease reach a significantly lower mean max heart rate (`thalach`: ~139 bpm) compared to healthy individuals (~158 bpm).
  * Welch's t-test confirms a statistically significant difference ($p < 10^{-4}$).
* **Age vs. Max Heart Rate:** 
  * Moderate negative correlation ($r \approx -0.39, p < 10^{-4}$). As age increases, maximum achievable heart rate decreases linearly.

### Dataset 2: Correlation vs. Causation
* **Naive Analysis:** 
  * Initially, one chemical (e.g., Benzene) displays a strong naive correlation with tumor malignancy, appearing to be the driver.
* **Controlled Analysis (Confounding Variable):** 
  * After controlling for `pollution_index` (filtering patients with similar exposure), the correlation for the misleading chemical drops to near zero, while the true chemical's association persists.
  * **Conclusion:** `pollution_index` acts as a confounding factor. Controlling for it reveals which chemical is the true causal factor versus the spurious variable.