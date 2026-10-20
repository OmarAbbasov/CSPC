import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# --- DATASET 1: HEART DISEASE ---
df_heart = pd.read_csv("heart.csv")

# 4. Is the effect real? (thalach in target=1 vs target=0)
thalach_healthy = df_heart[df_heart["target"] == 0]["thalach"]
thalach_disease = df_heart[df_heart["target"] == 1]["thalach"]

# Since thalach is approximately normal, we use Student's t-test / Welch's t-test
t_stat, p_val = stats.ttest_ind(thalach_healthy, thalach_disease, equal_var=False)
print("=== Heart Disease vs Max Heart Rate (thalach) ===")
print(f"Healthy mean thalach: {thalach_healthy.mean():.2f} ± {stats.sem(thalach_healthy):.2f}")
print(f"Disease mean thalach: {thalach_disease.mean():.2f} ± {stats.sem(thalach_disease):.2f}")
print(f"Welch's t-test: t={t_stat:.4f}, p-value={p_val:.4e}")

# Plot Means with Standard Error (SEM)
fig, ax = plt.subplots(figsize=(6, 5))
groups = ["Healthy (0)", "Disease (1)"]
means = [thalach_healthy.mean(), thalach_disease.mean()]
errors = [stats.sem(thalach_healthy), stats.sem(thalach_disease)]

ax.bar(groups, means, yerr=errors, capsize=5, color=["green", "red"], alpha=0.7)
ax.set_ylabel("Max Heart Rate (thalach)")
ax.set_title("Mean Max Heart Rate by Heart Disease Status (with SEM)")
plt.tight_layout()
plt.savefig("heart_rate_comparison.png")

# 5. Correlation between Age and Thalach
corr_pearson, p_pearson = stats.pearsonr(df_heart["age"], df_heart["thalach"])
print(f"\nAge vs Thalach Pearson correlation: r = {corr_pearson:.4f} (p = {p_pearson:.4e})")

plt.figure(figsize=(7, 5))
plt.scatter(df_heart["age"], df_heart["thalach"], alpha=0.6, color="purple")
m, b = np.polyfit(df_heart["age"], df_heart["thalach"], 1)
plt.plot(df_heart["age"], m*df_heart["age"] + b, color="black", linestyle="--")
plt.xlabel("Age")
plt.ylabel("Max Heart Rate (thalach)")
plt.title("Age vs Max Heart Rate")
plt.grid(True)
plt.tight_layout()
plt.savefig("age_vs_thalach.png")


# --- DATASET 2: CHEMICALS & CANCER ---
print("\n=== Chemical-Exposure Mystery ===")
df_chem = pd.read_csv("chemicals_cancer.csv")

# 6. Naive analysis
corr_benz = stats.pearsonr(df_chem["benzene"], df_chem["malignancy"])[0]
corr_cadm = stats.pearsonr(df_chem["cadmium"], df_chem["malignancy"])[0]

print(f"Naive correlation with Malignancy:")
print(f"  Benzene: {corr_benz:.4f}")
print(f"  Cadmium: {corr_cadm:.4f}")

# 7. Controlling for Confounding Variable (pollution_index / age)
# Filtering patients within a narrow pollution_index range
p_low, p_high = df_chem["pollution_index"].quantile(0.4), df_chem["pollution_index"].quantile(0.6)
df_filtered = df_chem[(df_chem["pollution_index"] >= p_low) & (df_chem["pollution_index"] <= p_high)]

corr_benz_adj = stats.pearsonr(df_filtered["benzene"], df_filtered["malignancy"])[0]
corr_cadm_adj = stats.pearsonr(df_filtered["cadmium"], df_filtered["malignancy"])[0]

print(f"\nAdjusted correlation (controlling for Pollution Index):")
print(f"  Benzene: {corr_benz_adj:.4f}")
print(f"  Cadmium: {corr_cadm_adj:.4f}")