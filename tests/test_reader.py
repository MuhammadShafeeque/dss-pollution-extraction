import pytest
import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
from pollution_extraction.core.data_reader import PollutionDataReader


@pytest.fixture
def sample_dataset(tmp_path):
    """Creates a temporary NetCDF file with mock NO2 data for testing."""
    file_path = tmp_path / "sample_no2.nc"

    time = pd.date_range("2006-01-01", periods=5)
    x = [10.0, 10.5]
    y = [50.0, 50.5]
    data = np.array(
        [
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
            [[2, 3], [4, 5]],
            [[6, 7], [8, 9]],
            [[1, 1], [1, 1]],
        ]
    )

    ds = xr.Dataset(
        {"no2_downscaled": (["time", "y", "x"], data)},
        coords={"time": time, "x": x, "y": y},
    )
    ds.to_netcdf(file_path)
    return file_path


@pytest.fixture
def reader(sample_dataset):
    """Returns a PollutionDataReader instance using the sample dataset."""
    return PollutionDataReader(file_path=sample_dataset)


def test_auto_detect_pollution_type(reader):
    assert reader.pollution_type == "no2"
    assert reader.variable_info["var_name"] == "no2_downscaled"


def test_get_basic_info(reader):
    info = reader.get_basic_info()

    assert isinstance(info, dict)
    assert info["pollution_type"] == "no2"
    assert info["spatial_dimensions"] == {"x": 2, "y": 2}
    assert info["total_time_steps"] == 5
    assert "time_range" in info
    assert "coordinate_system" in info


def test_get_data_summary(reader):
    summary = reader.get_data_summary()

    assert isinstance(summary, dict)
    assert "mean" in summary
    assert "max" in summary
    assert "min" in summary
    assert summary["count_valid"] > 0
    assert summary["missing_percentage"] == 0.0


def test_select_time_range(reader):
    subset = reader.select_time_range("2006-01-02", "2006-01-04")
    assert isinstance(subset, xr.Dataset)
    assert len(subset.time) == 3


def test_select_time_points(reader):
    subset = reader.select_time_points(["2006-01-01", "2006-01-05"])
    assert isinstance(subset, xr.Dataset)
    assert len(subset.time) == 2


def test_close_dataset(reader):
    reader.close()
    assert reader.dataset._file_obj is None or reader.dataset._file_obj.closed
