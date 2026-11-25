import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.inspection import permutation_importance


def adjusted_r2(r2: float, n: int, p: int) -> float:
    return float(1.0 - (1.0 - r2) * (n - 1) / (n - p - 1))


def main() -> None:
    os.makedirs("results", exist_ok=True)

    # A2. Clean and look
    df_raw = pd.read_csv("data/concrete.csv")
    dup_count = int(df_raw.duplicated().sum())
    missing_per_col = df_raw.isna().sum()

    df = df_raw.drop_duplicates().reset_index(drop=True)
    feature_cols = [
        "cement",
        "slag",
        "ash",
        "water",
        "superplastic",
        "coarse_agg",
        "fine_agg",
        "age",
    ]
    X = df[feature_cols].copy()
    y = df["strength"].copy()

    print("=== A2: CLEANING & SUMMARY ===")
    print(f"Raw shape: {df_raw.shape}, Duplicates removed: {dup_count}, Cleaned shape: {df.shape}")
    print("Missing values per column:\n", missing_per_col)
    print("\nDescribe cleaned dataset:\n", df.describe().round(2).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(X["cement"], y, alpha=0.45, s=16)
    axes[0].set_xlabel("Cement (kg/m^3)")
    axes[0].set_ylabel("Compressive Strength (MPa)")
    axes[0].set_title("Strength vs. Cement")
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(X["age"], y, alpha=0.45, s=16, color="tab:orange")
    axes[1].set_xlabel("Age (days)")
    axes[1].set_ylabel("Compressive Strength (MPa)")
    axes[1].set_title("Strength vs. Age")
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/A2_strength_relationships.png", dpi=180)
    plt.close()

    # A3. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print("\n=== A3: TRAIN / TEST SPLIT ===")
    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")

    # A4. Linear regression
    lr = LinearRegression(fit_intercept=True)
    lr.fit(X_train, y_train)

    y_train_pred = lr.predict(X_train)
    y_test_pred = lr.predict(X_test)

    n_train, p_lin = X_train.shape
    lin_table = pd.DataFrame([{
        "model": "LinearRegression",
        "p_features": p_lin,
        "train_rmse": float(np.sqrt(mean_squared_error(y_train, y_train_pred))),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, y_test_pred))),
        "test_mae": float(mean_absolute_error(y_test, y_test_pred)),
        "train_r2": float(r2_score(y_train, y_train_pred)),
        "train_adj_r2": adjusted_r2(r2_score(y_train, y_train_pred), n_train, p_lin),
        "test_r2": float(r2_score(y_test, y_test_pred)),
    }])
    lin_table.to_csv("results/A4_linear_metrics.csv", index=False)

    print("\n=== A4: LINEAR REGRESSION ===")
    print(f"Intercept (beta_0): {lr.intercept_:.4f} MPa")
    for col, coef in zip(feature_cols, lr.coef_):
        unit = "MPa/day" if col == "age" else "MPa/(kg/m^3)"
        print(f"  {col:14s}: {coef:+.6f} {unit}")
    print(lin_table.round(4).to_string(index=False))

    residuals_test = y_test - y_test_pred
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(X_test["age"], residuals_test, alpha=0.55, s=18)
    ax.set_xscale("log")
    ax.axhline(0.0, color="red", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Age (days, log scale)")
    ax.set_ylabel("Residual (True - Predicted, MPa)")
    ax.set_title("Test Residuals vs. Age (Log Scale)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/A4_residuals_vs_age.png", dpi=180)
    plt.close()

    # A5. Polynomial regression (degrees 1, 2, 3, 4)
    poly_rows = []
    for deg in [1, 2, 3, 4]:
        poly_model = make_pipeline(
            PolynomialFeatures(degree=deg, include_bias=False),
            StandardScaler(),
            LinearRegression(),
        )
        poly_model.fit(X_train, y_train)
        p_deg = int(poly_model[0].n_output_features_)
        tr_pred = poly_model.predict(X_train)
        te_pred = poly_model.predict(X_test)
        tr_r2 = float(r2_score(y_train, tr_pred))
        poly_rows.append({
            "degree": deg,
            "p_features": p_deg,
            "train_rmse": float(np.sqrt(mean_squared_error(y_train, tr_pred))),
            "test_rmse": float(np.sqrt(mean_squared_error(y_test, te_pred))),
            "train_r2": tr_r2,
            "train_adj_r2": adjusted_r2(tr_r2, n_train, p_deg),
            "test_r2": float(r2_score(y_test, te_pred)),
        })
    poly_df = pd.DataFrame(poly_rows)
    poly_df.to_csv("results/A5_polynomial_table.csv", index=False)
    print("\n=== A5: POLYNOMIAL REGRESSION ===")
    print(poly_df.round(4).to_string(index=False))

    # A6. Classification: KNN & Decision Tree (strength >= 40 MPa)
    y_class = (y >= 40.0).astype(int)
    print("\n=== A6: CLASSIFICATION TARGET (>= 40 MPa) ===")
    print("Class proportions:\n", y_class.value_counts(normalize=True).round(4))

    Xc_train, Xc_test, yc_train, yc_test = train_test_split(
        X, y_class, test_size=0.2, random_state=42, stratify=y_class
    )

    k_values = [1, 3, 5, 7, 15, 25, 51, 101]
    k_rows = []
    for k in k_values:
        knn_pipe = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))
        knn_pipe.fit(Xc_train, yc_train)
        k_rows.append({
            "K": k,
            "train_acc": float(knn_pipe.score(Xc_train, yc_train)),
            "test_acc": float(knn_pipe.score(Xc_test, yc_test)),
        })
    k_df = pd.DataFrame(k_rows)
    k_df.to_csv("results/A6_knn_sweep.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(k_df["K"], k_df["train_acc"], marker="o", label="Train accuracy")
    ax.plot(k_df["K"], k_df["test_acc"], marker="s", label="Test accuracy")
    ax.set_xscale("log")
    ax.set_xlabel("K (n_neighbors, log scale)")
    ax.set_ylabel("Accuracy")
    ax.set_title("KNN Accuracy vs. K (Scaled)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/A6_knn_sweep.png", dpi=180)
    plt.close()

    chosen_k = 7
    knn_scaled = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=chosen_k))
    knn_scaled.fit(Xc_train, yc_train)
    acc_scaled = float(knn_scaled.score(Xc_test, yc_test))

    knn_unscaled = KNeighborsClassifier(n_neighbors=chosen_k)
    knn_unscaled.fit(Xc_train, yc_train)
    acc_unscaled = float(knn_unscaled.score(Xc_test, yc_test))
    print(f"Chosen K={chosen_k}: test acc (scaled)={acc_scaled:.4f}, (unscaled)={acc_unscaled:.4f}")

    split_accs = []
    for rs in [0, 1, 2, 3, 4]:
        Xtr_s, Xte_s, ytr_s, yte_s = train_test_split(
            X, y_class, test_size=0.2, random_state=rs, stratify=y_class
        )
        m_s = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=chosen_k))
        m_s.fit(Xtr_s, ytr_s)
        split_accs.append({"random_state": rs, "test_acc": float(m_s.score(Xte_s, yte_s))})
    splits_df = pd.DataFrame(split_accs)
    splits_df.to_csv("results/A6_knn_splits.csv", index=False)
    print(
        f"5-split KNN (K={chosen_k}) test accuracies: "
        f"{[round(v, 4) for v in splits_df['test_acc']]}, "
        f"spread (max-min)={splits_df['test_acc'].max() - splits_df['test_acc'].min():.4f}"
    )

    depths = [1, 2, 3, 4, 5, 6, 8, None]
    tree_rows = []
    for d in depths:
        clf = DecisionTreeClassifier(
            criterion="entropy", max_depth=d, min_samples_leaf=1, random_state=0
        )
        clf.fit(Xc_train, yc_train)
        tree_rows.append({
            "max_depth": "None" if d is None else str(d),
            "depth": int(clf.get_depth()),
            "n_leaves": int(clf.get_n_leaves()),
            "train_acc": float(clf.score(Xc_train, yc_train)),
            "test_acc": float(clf.score(Xc_test, yc_test)),
        })
    tree_df = pd.DataFrame(tree_rows)
    tree_df.to_csv("results/A6_tree_sweep.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x_pos = np.arange(len(tree_df))
    ax.plot(x_pos, tree_df["train_acc"], marker="o", label="Train accuracy")
    ax.plot(x_pos, tree_df["test_acc"], marker="s", label="Test accuracy")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(tree_df["max_depth"])
    ax.set_xlabel("max_depth")
    ax.set_ylabel("Accuracy")
    ax.set_title("Decision Tree Accuracy vs. max_depth")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/A6_tree_sweep.png", dpi=180)
    plt.close()

    chosen_depth = 4
    tree_chosen = DecisionTreeClassifier(
        criterion="entropy", max_depth=chosen_depth, min_samples_leaf=1, random_state=0
    )
    tree_chosen.fit(Xc_train, yc_train)
    print(f"\nChosen Decision Tree (max_depth={chosen_depth}) top 2 levels:")
    print(export_text(tree_chosen, feature_names=list(X.columns), max_depth=2))

    # A7. Metrics that matter & threshold analysis
    def model_metrics_row(name: str, y_true: pd.Series, y_pred: np.ndarray, threshold: float = 0.5) -> dict:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return {
            "model": name,
            "threshold": threshold,
            "TP": int(tp),
            "FP": int(fp),
            "FN": int(fn),
            "TN": int(tn),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        }

    knn_pred = knn_scaled.predict(Xc_test)
    tree_pred = tree_chosen.predict(Xc_test)

    metrics_df = pd.DataFrame([
        model_metrics_row(f"KNN_K{chosen_k}", yc_test, knn_pred, threshold=0.5),
        model_metrics_row(f"DecisionTree_d{chosen_depth}", yc_test, tree_pred, threshold=0.5),
    ])
    metrics_df.to_csv("results/A_metrics.csv", index=False)
    print("\n=== A7: METRICS TABLE (results/A_metrics.csv) ===")
    print(metrics_df.round(4).to_string(index=False))

    proba_tree = tree_chosen.predict_proba(Xc_test)[:, 1]
    thresh_rows = []
    for t in [0.3, 0.5, 0.7, 0.9]:
        y_t = (proba_tree >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(yc_test, y_t).ravel()
        thresh_rows.append({
            "threshold": t,
            "TP": int(tp),
            "FP": int(fp),
            "FN": int(fn),
            "TN": int(tn),
            "precision": float(precision_score(yc_test, y_t, zero_division=0)),
            "recall": float(recall_score(yc_test, y_t, zero_division=0)),
        })
    thresh_df = pd.DataFrame(thresh_rows)
    thresh_df.to_csv("results/A7_thresholds_tree.csv", index=False)
    print("\n=== A7: DECISION TREE THRESHOLDS ===")
    print(thresh_df.round(4).to_string(index=False))

    # A8. Feature importance
    std_pipe = make_pipeline(StandardScaler(), LinearRegression())
    std_pipe.fit(X_train, y_train)
    std_coefs = pd.Series(std_pipe[-1].coef_, index=feature_cols)

    perm_knn = permutation_importance(
        knn_scaled, Xc_test, yc_test, scoring="f1", n_repeats=20, random_state=0
    )
    perm_tree = permutation_importance(
        tree_chosen, Xc_test, yc_test, scoring="f1", n_repeats=20, random_state=0
    )

    imp_df = pd.DataFrame({
        "feature": feature_cols,
        "raw_coef_reg": lr.coef_,
        "std_coef_reg": std_coefs.values,
        "tree_importance": tree_chosen.feature_importances_,
        "perm_f1_knn": perm_knn.importances_mean,
        "perm_f1_tree": perm_tree.importances_mean,
    })
    imp_df.to_csv("results/A8_feature_importance.csv", index=False)
    print("\n=== A8: FEATURE IMPORTANCE TABLE ===")
    print(imp_df.round(4).to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    imp_df.sort_values("std_coef_reg").plot(
        x="feature", y="std_coef_reg", kind="barh", ax=axes[0], legend=False
    )
    axes[0].set_title("Standardized Coef (Reg)")

    imp_df.sort_values("tree_importance").plot(
        x="feature", y="tree_importance", kind="barh", ax=axes[1], legend=False, color="tab:green"
    )
    axes[1].set_title("Tree Feature Importance")

    imp_df.sort_values("perm_f1_knn").plot(
        x="feature", y="perm_f1_knn", kind="barh", ax=axes[2], legend=False, color="tab:purple"
    )
    axes[2].set_title("Permutation Importance (KNN, F1)")

    plt.tight_layout()
    plt.savefig("results/A8_feature_importance.png", dpi=180)
    plt.close()

    # Bonus: Physics-informed features (log(age) and w/b)
    X_log_age = X_train.copy()
    X_log_age["age"] = np.log(X_log_age["age"])
    Xte_log_age = X_test.copy()
    Xte_log_age["age"] = np.log(Xte_log_age["age"])

    m_log_age = LinearRegression().fit(X_log_age, y_train)
    r2_log_age = float(r2_score(y_test, m_log_age.predict(Xte_log_age)))

    binder_train = X_train["cement"] + X_train["slag"] + X_train["ash"]
    binder_test = X_test["cement"] + X_test["slag"] + X_test["ash"]

    X_wb = X_log_age.drop(columns=["water"]).copy()
    X_wb["w_over_b"] = X_train["water"] / binder_train.replace(0, np.nan)
    Xte_wb = Xte_log_age.drop(columns=["water"]).copy()
    Xte_wb["w_over_b"] = X_test["water"] / binder_test.replace(0, np.nan)

    valid_tr = X_wb.notna().all(axis=1)
    valid_te = Xte_wb.notna().all(axis=1)
    m_wb = LinearRegression().fit(X_wb[valid_tr], y_train[valid_tr])
    r2_wb = float(r2_score(y_test[valid_te], m_wb.predict(Xte_wb[valid_te])))

    bonus_df = pd.DataFrame([
        {"variant": "log(age) only", "p_features": X_log_age.shape[1], "test_r2": r2_log_age},
        {"variant": "log(age) + w/b", "p_features": X_wb.shape[1], "test_r2": r2_wb},
    ])
    bonus_df.to_csv("results/A_bonus_physics.csv", index=False)
    print("\n=== BONUS (PHYSICS FEATURES) ===")
    print(bonus_df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()