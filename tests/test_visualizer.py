from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.figure
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pollution_extraction.core.data_visualizer import DataVisualizer

if TYPE_CHECKING:
    pass  # No special type imports needed


@pytest.fixture
def sample_dataset() -> xr.Dataset:
    """Create a small mock dataset for visualization tests.

    Returns a dataset with monthly NO2 data."""
    time = pd.date_range("2006-01-01", periods=12, freq="MS")
    x = np.linspace(0, 1, 4)
    y = np.linspace(0, 1, 3)
    data = np.random.rand(len(time), len(y), len(x))

    ds = xr.Dataset(
        {"no2_downscaled": (["time", "y", "x"], data)},
        coords={"time": time, "x": x, "y": y},
    )
    return ds


@pytest.fixture
def visualizer(
    sample_dataset: xr.Dataset,
) -> DataVisualizer:
    """Returns a DataVisualizer instance using the sample dataset."""
    return DataVisualizer(
        dataset=sample_dataset,
        pollution_variable="no2_downscaled",
        pollution_type="no2",
    )


def test_plot_spatial_map(
    visualizer: DataVisualizer,
    tmp_path: Path,
) -> None:
    fig = visualizer.plot_spatial_map(time_index=0, save_path=tmp_path / "map.png")
    assert isinstance(fig, matplotlib.figure.Figure)
    assert (tmp_path / "map.png").exists()


def test_plot_time_series(
    visualizer: DataVisualizer,
    tmp_path: Path,
) -> None:
    fig = visualizer.plot_time_series(save_path=tmp_path / "ts.png")
    assert isinstance(fig, matplotlib.figure.Figure)
    assert (tmp_path / "ts.png").exists()


def test_plot_distribution(
    visualizer: DataVisualizer,
    tmp_path: Path,
) -> None:
    fig = visualizer.plot_distribution(save_path=tmp_path / "dist.png")
    assert isinstance(fig, matplotlib.figure.Figure)
    assert (tmp_path / "dist.png").exists()


def test_plot_spatial_statistics(
    visualizer: DataVisualizer,
    tmp_path: Path,
) -> None:
    fig = visualizer.plot_spatial_statistics(
        statistic="mean", save_path=tmp_path / "stat.png"
    )
    assert isinstance(fig, matplotlib.figure.Figure)
    assert (tmp_path / "stat.png").exists()


def test_plot_seasonal_cycle(
    visualizer: DataVisualizer,
    tmp_path: Path,
) -> None:
    fig = visualizer.plot_seasonal_cycle(save_path=tmp_path / "season.png")
    assert isinstance(fig, matplotlib.figure.Figure)
    assert (tmp_path / "season.png").exists()


def test_plot_comparison(
    sample_dataset: xr.Dataset,
    tmp_path: Path,
) -> None:
    # Create a second dataset with a small offset
    modified_ds = sample_dataset.copy(deep=True)
    modified_ds["no2_downscaled"] += 0.5

    visualizer = DataVisualizer(
        dataset=sample_dataset,
        pollution_variable="no2_downscaled",
        pollution_type="no2",
    )

    fig = visualizer.plot_comparison(
        other_dataset=modified_ds,
        other_variable="no2_downscaled",
        comparison_type="difference",
        time_index=0,
        save_path=tmp_path / "compare_diff.png",
    )

    assert isinstance(fig, matplotlib.figure.Figure)
    assert (tmp_path / "compare_diff.png").exists()
