import pytest
import xarray as xr
import numpy as np
import pandas as pd
from pollution_extraction.core.temporal_aggregator import TemporalAggregator


@pytest.fixture
def sample_dataset():
    """Create a small mock dataset with daily data for one year."""
    time = pd.date_range("2020-01-01", "2020-12-31", freq="D")
    x = np.linspace(0, 1, 2)
    y = np.linspace(0, 1, 2)
    data = np.random.rand(len(time), len(y), len(x))

    ds = xr.Dataset(
        {"no2_downscaled": (["time", "y", "x"], data)},
        coords={"time": time, "x": x, "y": y},
    )
    return ds


@pytest.fixture
def aggregator(sample_dataset):
    return TemporalAggregator(
        dataset=sample_dataset, pollution_variable="no2_downscaled"
    )


def test_monthly_aggregation_mean(aggregator):
    monthly = aggregator.monthly_aggregation(method="mean")
    assert isinstance(monthly, xr.Dataset)
    assert monthly.time.size == 0  # No time coordinate expected
    assert monthly["no2_downscaled"].size == 12


def test_annual_aggregation_sum(aggregator):
    annual = aggregator.annual_aggregation(method="sum")
    assert isinstance(annual, xr.Dataset)
    assert annual["no2_downscaled"].size == 1


def test_seasonal_aggregation_std(aggregator):
    seasonal = aggregator.seasonal_aggregation(method="std")
    assert isinstance(seasonal, xr.Dataset)
    assert set(seasonal["season"].values).issubset(
        {"winter", "spring", "summer", "autumn"}
    )


def test_custom_time_aggregation(aggregator):
    periods = [("2020-01-01", "2020-03-31"), ("2020-04-01", "2020-06-30")]
    result = aggregator.custom_time_aggregation(
        periods, method="mean", period_names=["Q1", "Q2"]
    )
    assert "period" in result.coords
    assert result["period"].size == 2


def test_rolling_aggregation(aggregator):
    rolling = aggregator.rolling_aggregation(window=7, method="mean")
    assert isinstance(rolling, xr.Dataset)
    assert "time" in rolling.dims


def test_time_groupby_aggregation_quarterly(aggregator):
    grouped = aggregator.time_groupby_aggregation(groupby_freq="Q", method="mean")
    assert isinstance(grouped, xr.Dataset)
    assert grouped["no2_downscaled"].shape[0] == 4  # Q1–Q4


def test_get_time_before_date(aggregator):
    subset = aggregator.get_time_before_date(reference_date="2020-12-31", days_before=7)
    assert isinstance(subset, xr.Dataset)
    assert subset.time.size == 8  # includes the ref date


def test_aggregate_multiple_periods(aggregator):
    config = [
        {"type": "monthly", "method": "mean", "name": "monthly_avg"},
        {"type": "annual", "method": "sum", "name": "annual_sum"},
        {"type": "seasonal", "method": "max", "name": "seasonal_max"},
        {
            "type": "custom",
            "method": "mean",
            "time_periods": [("2020-01-01", "2020-06-30")],
            "period_names": ["first_half"],
            "name": "custom_half",
        },
    ]
    results = aggregator.aggregate_multiple_periods(config)
    assert isinstance(results, dict)
    assert "monthly_avg" in results
    assert "annual_sum" in results
    assert "seasonal_max" in results
    assert "custom_half" in results
