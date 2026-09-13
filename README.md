# CSPC Computer Science for Physics and Chemistry

My coursework repository. Each practical is under PW/Lab /.

## Setup

Create the environment for a given lab:

```bash
conda env create -f PW<n>/Lab\ <X>/environment.yml
conda activate cspc ```

PW1 - Lab A: Reproducible Foundations
What I built:
Created the CSPC repository structure, configured conda environment, integrated Git/GitHub workflow, written unit tests with pytest, and benchmarked NumPy against Python loops.

Speed comparison (loop vs NumPy):

Loop: 3.0120 s

NumPy: 0.0004 s

Speed-up: 8466.96x faster

Tests: All passing? Yes

Conclusion:
NumPy vectorization dramatically increases computational performance compared to pure Python loops.