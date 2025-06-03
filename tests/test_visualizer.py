import pytest
import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
from pollution_extraction.core.data_visualizer import DataVisualizer
import matplotlib.figure


@pytest.fixture
def sample_dataset():
    """Creates a small mock dataset with monthly data for visualization tests."""
    time = pd.date_range("2006-01-01", periods=12, freq="MS")  # 12 months
    x = np.linspace(0, 1, 4)
    y = np.linspace(0, 1, 3)
    data = np.random.rand(len(time), len(y), len(x))

    ds = xr.Dataset(
        {"no2_downscaled": (["time", "y", "x"], data)},
        coords={"time": time, "x": x, "y": y},
    )
    return ds


@pytest.fixture
def visualizer(sample_dataset):
    """Returns a DataVisualizer instance using the sample dataset."""
    return DataVisualizer(
        dataset=sample_dataset,
        pollution_variable="no2_downscaled",
        pollution_type="no2",
    )


def test_plot_spatial_map(visualizer, tmp_path):
    fig = visualizer.plot_spatial_map(time_index=0, save_path=tmp_path / "map.png")
    assert isinstance(fig, matplotlib.figure.Figure)
    assert (tmp_path / "map.png").exists()


def test_plot_time_series(visualizer, tmp_path):
    fig = visualizer.plot_time_series(save_path=tmp_path / "ts.png")
    assert isinstance(fig, matplotlib.figure.Figure)
    assert (tmp_path / "ts.png").exists()


def test_plot_distribution(visualizer, tmp_path):
    fig = visualizer.plot_distribution(save_path=tmp_path / "dist.png")
    assert isinstance(fig, matplotlib.figure.Figure)
    assert (tmp_path / "dist.png").exists()


def test_plot_spatial_statistics(visualizer, tmp_path):
    fig = visualizer.plot_spatial_statistics(
        statistic="mean", save_path=tmp_path / "stat.png"
    )
    assert isinstance(fig, matplotlib.figure.Figure)
    assert (tmp_path / "stat.png").exists()


def test_plot_seasonal_cycle(visualizer, tmp_path):
    fig = visualizer.plot_seasonal_cycle(save_path=tmp_path / "season.png")
    assert isinstance(fig, matplotlib.figure.Figure)
    assert (tmp_path / "season.png").exists()


def test_plot_comparison(sample_dataset, tmp_path):
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


def test_create_animation(sample_dataset, tmp_path):
    visualizer = DataVisualizer(
        dataset=sample_dataset,
        pollution_variable="no2_downscaled",
        pollution_type="no2",
    )

    output_path = tmp_path / "animation.gif"
    visualizer.create_animation(output_path=output_path, time_step=1, fps=2, dpi=80)

    assert output_path.exists()
    assert output_path.suffix == ".gif"
