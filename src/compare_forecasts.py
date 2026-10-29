
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"

BASELINE_PATH = REPORTS_DIR / "spy_variance_forecasts.csv"
RF_PATH = REPORTS_DIR / "spy_random_forest_forecasts.csv"


def load_forecasts() -> pd.DataFrame:
    baseline = pd.read_csv(
        BASELINE_PATH,
        parse_dates=["Date"],
        index_col="Date",
    )

    random_forest = pd.read_csv(
        RF_PATH,
        parse_dates=["Date"],
        index_col="Date",
    )

    if baseline.index.has_duplicates or random_forest.index.has_duplicates:
        raise ValueError("Duplicate forecast dates detected.")

    # Keep only dates present in both files.
    common = baseline.join(
        random_forest[["random_forest", "realized_variance"]].rename(
            columns={"realized_variance": "realized_variance_rf"}
        ),
        how="inner",
    ).sort_index()

    if common.empty:
        raise ValueError("No overlapping forecast dates found.")

    # Confirm both files use the same realized squared-return proxy.
    if not np.allclose(
        common["realized_variance"],
        common["realized_variance_rf"],
        rtol=1e-8,
        atol=1e-12,
    ):
        raise ValueError(
            "Realized variance values disagree on common dates."
        )

    common = common.drop(columns=["realized_variance_rf"])

    model_columns = ["garch", "ewma", "historical", "random_forest"]

    if not np.isfinite(common[model_columns].to_numpy()).all():
        raise ValueError("Forecasts contain non-finite values.")

    if (common[model_columns] <= 0).any().any():
        raise ValueError("Variance forecasts must be positive.")

    return common


def evaluate(common: pd.DataFrame) -> pd.DataFrame:
    actual = common["realized_variance"].to_numpy()
    results = {}

    for model_name in ["garch", "ewma", "historical", "random_forest"]:
        prediction = common[model_name].to_numpy()

        qlike = np.log(prediction) + actual / prediction
        mse = np.mean((actual - prediction) ** 2)

        results[model_name] = {
            "Mean QLIKE": float(np.mean(qlike)),
            "MSE": float(mse),
        }

    return pd.DataFrame(results).T.sort_values("Mean QLIKE")


def main() -> None:
    common = load_forecasts()
    metrics = evaluate(common)

    print("Fair out-of-sample variance forecast comparison")
    print(f"Common observations: {len(common):,}")
    print(f"Common test begins: {common.index.min().date()}")
    print(f"Common test ends: {common.index.max().date()}")

    print("\nMetrics on identical dates (lower is better):")
    print(metrics.to_string(float_format=lambda x: f"{x:.9e}"))

    output_path = REPORTS_DIR / "spy_model_comparison.csv"
    metrics.to_csv(output_path, index_label="model")
    print(f"\nSaved comparison to: {output_path}")


if __name__ == "__main__":
    main()
