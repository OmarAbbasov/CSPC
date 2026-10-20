import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# 1. Load Data
df = pd.read_csv("heart.csv")
print(f"Dataset shape: {df.shape}")
print(df.head())

continuous_cols = ["age", "chol", "trestbps", "thalach"]

# 2. Visualize Distributions
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.flatten()

for i, col in enumerate(continuous_cols):
    axes[i].hist(df[col], bins=20, color="skyblue", edgecolor="black", density=True, alpha=0.6)
    
    # Fit normal distribution overlay
    mu, std = df[col].mean(), df[col].std()
    xmin, xmax = axes[i].get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mu, std)
    axes[i].plot(x, p, 'k', linewidth=2, label="Normal fit")
    
    axes[i].set_title(f"Distribution of {col}")
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Density")
    axes[i].legend()

plt.tight_layout()
plt.savefig("session1_distributions.png")
print("Saved session1_distributions.png")

# 3. Normality Testing (Shapiro-Wilk test)
print("\n--- Normality Verdicts (Shapiro-Wilk Test) ---")
for col in continuous_cols:
    stat, p_val = stats.shapiro(df[col])
    is_normal = p_val > 0.05
    verdict = "Normal" if is_normal else "Non-Normal"
    print(f"{col:10s}: W={stat:.4f}, p-value={p_val:.4e} -> {verdict}")