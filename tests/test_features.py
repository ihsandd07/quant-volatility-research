

import numpy as np
import pandas as pd
import pytest

from src.features import calculate_returns


def test_simple_returns_are_calculated_correctly():
    prices = pd.DataFrame(
        {"ETF_A": [100.0, 110.0, 99.0]},
        index=pd.date_range("2024-01-01", periods=3),
    )

    simple, log_returns = calculate_returns(prices)

    assert len(simple) == 2
    assert simple["ETF_A"].iloc[0] == pytest.approx(0.10)
    assert simple["ETF_A"].iloc[1] == pytest.approx(-0.10)

    assert log_returns["ETF_A"].iloc[0] == pytest.approx(
        np.log(1.10)
    )
    assert log_returns["ETF_A"].iloc[1] == pytest.approx(
        np.log(0.90)
    )


def test_returns_preserve_asset_columns():
    prices = pd.DataFrame(
        {
            "SPY": [100.0, 101.0, 102.0],
            "AGG": [100.0, 99.0, 100.0],
        }
    )

    simple, log_returns = calculate_returns(prices)

    assert list(simple.columns) == ["SPY", "AGG"]
    assert list(log_returns.columns) == ["SPY", "AGG"]


def test_returns_reject_nonpositive_prices():
    prices = pd.DataFrame({"ETF_A": [100.0, 0.0]})

    with pytest.raises(ValueError, match="positive"):
        calculate_returns(prices)


def test_returns_reject_empty_prices():
    with pytest.raises(ValueError, match="empty"):
        calculate_returns(pd.DataFrame())


def test_returns_contain_no_invalid_values():
    prices = pd.DataFrame(
        {"ETF_A": [100.0, 102.0, 101.0, 105.0]}
    )

    simple, log_returns = calculate_returns(prices)

    assert np.isfinite(simple.to_numpy()).all()
    assert np.isfinite(log_returns.to_numpy()).all()
