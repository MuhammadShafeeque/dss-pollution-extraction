from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import Point, Polygon

from pollution_extraction.core.spatial_extractor import SpatialExtractor


@pytest.fixture
def mock_dataset() -> xr.Dataset:
    """Create a mock xarray.Dataset with spatial and temporal dimensions."""
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
def point_geodata() -> gpd.GeoDataFrame:
    """GeoDataFrame with two points in EPSG:4326."""
    geometry = [Point(10.2, 50.3), Point(10.4, 50.4)]
    gdf = gpd.GeoDataFrame({"name": ["A", "B"]}, geometry=geometry, crs="EPSG:4326")
    return gdf


@pytest.fixture
def polygon_geodata() -> gpd.GeoDataFrame:
    """GeoDataFrame with one square polygon."""
    polygon = Polygon(
        [(10.1, 50.1), (10.6, 50.1), (10.6, 50.6), (10.1, 50.6), (10.1, 50.1)]
    )
    gdf = gpd.GeoDataFrame(
        {"region": ["TestRegion"]}, geometry=[polygon], crs="EPSG:4326"
    )
    return gdf


@pytest.fixture
def extractor(mock_dataset: xr.Dataset) -> SpatialExtractor:
    """Returns a SpatialExtractor instance using the mock dataset."""
    return SpatialExtractor(dataset=mock_dataset, pollution_variable="no2_downscaled")


def test_constructor_with_valid_coords(extractor: SpatialExtractor) -> None:
    assert "x" in extractor.dataset.coords
    assert "y" in extractor.dataset.coords


def test_extract_points_from_geodata(
    extractor: SpatialExtractor, point_geodata: gpd.GeoDataFrame
) -> None:
    result = extractor.extract_points(point_geodata)
    assert isinstance(result, xr.Dataset)
    assert result.dims["location_id"] == 2
    assert "location_name" in result.coords


def test_extract_points_from_list(extractor: SpatialExtractor) -> None:
    points = [(10.2, 50.3), (10.5, 50.5)]
    result = extractor.extract_points(points)
    assert isinstance(result, xr.Dataset)
    assert result.dims["location_id"] == 2


def test_extract_polygons(
    extractor: SpatialExtractor, polygon_geodata: gpd.GeoDataFrame
) -> None:
    result = extractor.extract_polygons(polygon_geodata)
    assert isinstance(result, xr.Dataset)
    assert result.dims["polygon_id"] == 1
    assert "polygon_region" in result.coords


def test_spatial_subset_bbox(extractor: SpatialExtractor) -> None:
    bounds = {"minx": 10.1, "maxx": 10.6, "miny": 50.2, "maxy": 50.7}
    subset = extractor.spatial_subset(bounds)
    assert isinstance(subset, xr.Dataset)
    assert subset.x.size <= extractor.dataset.x.size
    assert subset.y.size <= extractor.dataset.y.size
