
import numpy as np
import pandas as pd

from src.portfolio_backtest import (
    ASSETS,
    MAX_WEIGHT,
    TARGET_VOL,
    get_weights,
)


def make_sample_returns(seed=42, n=252):
    """Generate deterministic sample returns for portfolio tests."""
    rng = np.random.default_rng(seed)
    data = rng.normal(0.0003, 0.01, size=(n, len(ASSETS)))
    return pd.DataFrame(
        data,
        columns=ASSETS,
        index=pd.bdate_range("2023-01-02", periods=n),
    )


def test_equal_weight_is_long_only_and_fully_invested():
    history = make_sample_returns()
    weights = get_weights(history, "equal_weight")

    assert np.all(weights >= -1e-10)
    assert np.isclose(weights.sum(), 1.0)
    assert np.all(weights <= MAX_WEIGHT + 1e-8)


def test_min_variance_respects_weight_constraints():
    history = make_sample_returns()
    weights = get_weights(history, "min_variance")

    assert np.all(weights >= -1e-8)
    assert np.all(weights <= MAX_WEIGHT + 1e-8)
    assert np.isclose(weights.sum(), 1.0, atol=1e-6)


def test_volatility_target_never_uses_leverage():
    history = make_sample_returns()
    weights = get_weights(history, "volatility_target")

    assert np.all(weights >= -1e-8)
    assert np.all(weights <= MAX_WEIGHT + 1e-8)
    assert weights.sum() <= 1.0 + 1e-8
    assert weights.sum() >= -1e-8


def test_spy_volatility_overlay_requires_valid_forecast():
    history = make_sample_returns()

    for invalid_forecast in (None, 0.0, -0.1, np.nan, np.inf):
        try:
            get_weights(
                history,
                "spy_vol_target",
                rf_variance=invalid_forecast,
            )
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"Expected ValueError for forecast {invalid_forecast}"
            )


def test_spy_volatility_overlay_respects_exposure_constraints():
    history = make_sample_returns()
    forecast_variance = 0.0001

    weights = get_weights(
        history,
        "spy_vol_target",
        rf_variance=forecast_variance,
    )

    assert np.all(weights >= -1e-8)
    assert np.all(weights <= MAX_WEIGHT + 1e-8)
    assert weights.sum() <= 1.0 + 1e-8
