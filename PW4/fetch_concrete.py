import os
import pandas as pd
from ucimlrepo import fetch_ucirepo


def main() -> None:
    os.makedirs("data", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    concrete = fetch_ucirepo(id=165)

    print("=== CONCRETE METADATA ===")
    for k, v in concrete.metadata.items():
        print(f"{k}: {v}")

    print("\n=== CONCRETE VARIABLES ===")
    print(concrete.variables.to_string(index=False))

    X = concrete.data.features.copy()
    y_df = concrete.data.targets.copy()

    expected_cols = [
        "cement",
        "slag",
        "ash",
        "water",
        "superplastic",
        "coarse_agg",
        "fine_agg",
        "age",
    ]
    assert len(X.columns) == len(expected_cols), "Unexpected number of feature columns"
    X.columns = expected_cols

    y = y_df.iloc[:, 0].rename("strength")

    df = X.assign(strength=y)
    df.to_csv("data/concrete.csv", index=False)
    print("\nSaved data/concrete.csv with shape:", df.shape)


if __name__ == "__main__":
    main()