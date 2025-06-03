"""Test fixtures for pollution_extraction test suite.

This module contains pytest fixtures used across multiple test modules.
"""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import Point, Polygon


@pytest.fixture
def mock_dataset():
    """Mock pollution dataset with time, x, y dimensions and CRS."""
    time = pd.date_range("2020-01-01", periods=3)
    x = np.linspace(10, 11, 4)
    y = np.linspace(50, 51, 3)
    data = np.random.rand(len(time), len(y), len(x))

    ds = xr.Dataset(
        {"no2_downscaled": (["time", "y", "x"], data)},
        coords={"time": time, "x": x, "y": y},
    )
    ds.rio.write_crs("EPSG:4326", inplace=True)
    return ds


@pytest.fixture
def point_geodata():
    """GeoDataFrame with two points in EPSG:4326."""
    geometry = [Point(10.2, 50.3), Point(10.4, 50.4)]
    gdf = gpd.GeoDataFrame({"name": ["A", "B"]}, geometry=geometry, crs="EPSG:4326")
    return gdf


@pytest.fixture
def polygon_geodata():
    """GeoDataFrame with one square polygon."""
    polygon = Polygon(
        [(10.1, 50.1), (10.6, 50.1), (10.6, 50.6), (10.1, 50.6), (10.1, 50.1)]
    )
    gdf = gpd.GeoDataFrame(
        {"region": ["TestRegion"]}, geometry=[polygon], crs="EPSG:4326"
    )
    return gdf
