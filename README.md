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

  # Practical Work 4 — Machine Learning: Supervised and Unsupervised

## Overview & One-Command Reproduction

This directory implements the end-to-end supervised and unsupervised machine-learning workflows from Practical Work 4:
- **Session A (`fetch_concrete.py`, `sessionA.py`):** Regression, classification, model evaluation, threshold selection, feature importance, and physics-informed feature engineering on the UCI Concrete Compressive Strength dataset (Yeh, 1998; UCI ID 165).
- **Session B (`sessionB.py`):** Unsupervised dimensionality reduction (PCA) and clustering (K-Means) on the UCI Wine dataset (Forina et al., 178 samples, 13 chemical measurements), compared against true grape cultivars.
- **Automation (`Snakefile`):** Reproduces all datasets, tables, and figures with a single command from `PW4/`:

```bash
conda activate cspc
snakemake --cores 1

# Practical Work 5 — Signal Processing: Where Was the Earthquake, and How Strong?

## How to Run and Reproduce

From `PW5/` with the `cspc` environment active:

    conda activate cspc
    python run_steps_1_to_8.py
    snakemake --cores 1

---

## 1. Record Overview (`seismogram_A.csv`)

- **Sampling & channels:** Three ground-velocity channels (`Z` vertical, `N` north-south, `E` east-west) in `m/s` (plotted in `mm/s`), sampled at `fs = 100 Hz` (`30,000` samples over `~300 s`), with wave speeds `v_P = 6.2 km/s` and `v_S = 3.6 km/s`.
- **Channel behavior & quiet window (`results/fig1_channels.png`, `results/fig1_zoom_60s.png`):** The first, weaker shaking (P-wave) appears most clearly on the vertical channel `Z`, whereas the later, stronger shaking (S-wave and surface waves) is largest on the horizontal channels `N` and `E`. Zooming in on the first `60 s` shows that the station is not completely still before the earthquake: low-level background seismic noise is continuously present.

![Seismogram A Channels](results/fig1_channels.png)

![First 60 s Zoom](results/fig1_zoom_60s.png)

---

## 2. Frequency Spectrum of `Z`

- **Index vs. `fftfreq` and symmetry (`results/fig2a_spectrum_index_vs_freq.png`):** Plotting against array index only shows sample positions (`0..N-1`), whereas `np.fft.fftfreq` maps each bin to physical frequency in `Hz` (`-fs/2` to `+fs/2`). Because the input seismogram is real-valued, its Fourier transform is conjugate-symmetric around `0 Hz`, so the amplitude spectrum is mirrored across positive and negative frequencies.
- **Quiet window vs. whole record (`results/fig2b_spectrum_loglog.png`):** On a log-log scale at positive frequencies, the quiet first `60 s` reveals background noise peaking in the low-frequency microseism band (around `0.1–0.5 Hz`, driven by ocean wave activity), while the earthquake adds the largest share of energy in the `~0.5–10 Hz` band.

![Spectrum Index vs Frequency](results/fig2a_spectrum_index_vs_freq.png)

![Log-Log Spectrum](results/fig2b_spectrum_loglog.png)

---

## 3. Inverse FFT and Manual Band-Pass Filter (`0.5–10 Hz`)

- **Round-trip check (`np.allclose(np.fft.ifft(np.fft.fft(Z)).real, Z)`):** True — transforming to frequency space and back recovers the original signal, which is why zeroing unwanted frequency bins and applying `ifft` works for frequency-domain filtering.
- **Cut-off choice (`a = 0.5 Hz`, `b = 10 Hz`):** `0.5 Hz` suppresses dominant low-frequency microseism/background drift, while `10 Hz` removes high-frequency noise above the main earthquake signal band.
- **Removed signal (`results/fig3c_removed_z.png`):** Plotting `z - z_bp` shows the out-of-band background noise and low/high-frequency components stripped outside `0.5–10 Hz`.

![Removed Signal](results/fig3c_removed_z.png)

---

## 4. Signal Envelope and Window Length (`results/fig4_envelope_windows.png`)

- Using a `0.5 s` moving-average window on `|x|` gives a clean envelope for filtered `Z` and total horizontal motion $\sqrt{N^2 + E^2}$.
- **Too short (`0.05 s`):** Too much ripple/noise remains, making threshold crossings and peaks jittery.
- **Too long (`5 s`):** The envelope smears across time and delays the apparent arrival and peak timing.

![Envelope Window Comparison](results/fig4_envelope_windows.png)

---

## 5. Arrival Times vs. Peak Shaking (`results/fig5_arrivals.png`)

- On `seismogram_A.csv`, `t_P` (`5%` of filtered `Z` envelope max) arrives first on `Z` (`108.42 s`), followed by `t_S` (`10%` of filtered horizontal envelope max after `t_P + 2 s` at `144.08 s`), while the horizontal envelope peak (`find_peaks`) occurs later (`152.03 s`) during stronger wave trains.
- **Why peak time gives the wrong distance:** An arrival time marks when a wave front begins traveling from the source, whereas peak amplitude occurs later due to wave interference, source rupture/attenuation dynamics, and windowing; using peak time inflates `t_S - t_P` and overestimates distance `R`.

![Arrival Times](results/fig5_arrivals.png)

---

## 6. Distance `R` and Origin Time

- Using $R = \frac{t_S - t_P}{\frac{1}{v_S} - \frac{1}{v_P}}$ with $v_P = 6.2\text{ km/s}$ and $v_S = 3.6\text{ km/s}$, `seismogram_A.csv` gives `R = 306.13 km` and origin time `t_origin = t_P - R / v_P = 59.05 s`.
- **Sensitivity to `1 s` uncertainty in `t_S`:** Since $\left(\frac{1}{3.6} - \frac{1}{6.2}\right)^{-1} \approx 8.58\text{ km/s}$, a `1 s` shift in `t_S` changes `R` by about `8.58 km`.

---

## 7. Spectrogram (`results/fig7_spectrogram.png`)

- The spectrogram (`nperseg=256`, up to `10 Hz`) resolves P-wave energy, stronger S-wave energy, and later, slower surface-wave energy over time. Higher frequencies arrive earlier with body waves (`P` and `S`), while lower frequencies persist into the later, slower surface waves.
- Unlike a single whole-record spectrum (Step 2), the spectrogram shows **when** each frequency band arrives rather than averaging frequency content across the entire record.

![Spectrogram](results/fig7_spectrogram.png)

---

## 8. PGV and Magnitude

- From raw horizontal velocity $\sqrt{N^2 + E^2}$ converted to `cm/s`, `seismogram_A.csv` has `PGV = 1.5105 cm/s` and magnitude `M = 6.66` via $M = \frac{\log_{10}(\text{PGV}[\text{cm/s}]) + \log_{10}(R[\text{km}]) + 1.0}{0.55}$.
- **Why $\log_{10} R$ is included:** Ground motion decays with distance (`R`); omitting $\log_{10} R$ would underestimate the magnitude of the same earthquake if recorded twice as far away.
- **PGV increase for `+1.0` magnitude unit at fixed `R`:** Increasing `M` by `1.0` increases $\log_{10}(\text{PGV})$ by `0.55`, so `PGV` increases by a factor of $10^{0.55} \approx 3.55\times$.

---

## 9. Multi-Event Comparison (`results/events.csv`, `results/report.txt`)

| Event | `t_P` (s) | `t_S` (s) | `R` (km) | Origin Time (s) | `PGV` (cm/s) | Magnitude `M` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `seismogram_A` | 108.42 | 144.08 | 306.13 | 59.05 | 1.5105 | 6.66 |
| `seismogram_B` | 84.25 | 102.43 | 156.07 | 59.08 | 0.5711 | 5.36 |

- **Closer earthquake:** `seismogram_B` (`156.07 km` vs. `306.13 km` for `seismogram_A`).
- **Stronger earthquake:** `seismogram_A` (`M = 6.66` vs. `M = 5.36` for `seismogram_B`).

---

## 10. Snakemake Automation

- **(a) Delete `results/` and run `snakemake --cores 1`:** Creates `results/`, runs `analyse` (`data/seismogram_A.csv`, `data/seismogram_B.csv`, `analyse.py` -> `results/events.csv`), and then runs `report` (`results/events.csv`, `report.py` -> `results/report.txt`).
- **(b) Run `snakemake --cores 1` again:** Reports nothing to be done (`Nothing to be done`), because all outputs already exist and are newer than their inputs.
- **(c) Change only wording in `report.py` and run again:** Only `report` (and `all`) reruns; `analyse` is skipped because its inputs and `results/events.csv` are unchanged.
- **(d) Change cut-off `b` in `analyse.py` and run again:** `analyse` reruns because `analyse.py` changed, and `report` also reruns because `results/events.csv` (its input) was updated. This dependency cascade is desirable so downstream summaries stay consistent with updated upstream analysis parameters.