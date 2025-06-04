"""Base classes and utilities for data export."""

import logging
from enum import Enum
from pathlib import Path
from typing import Literal

import xarray as xr

logger = logging.getLogger(__name__)


# Format list type definitions
TabularFormatList = list[Literal["csv", "json"]]
SpatialFormatList = list[Literal["geojson", "shapefile", "csv"]]
RasterFormatList = list[Literal["geotiff", "netcdf"]]


class ExportFormat(str, Enum):
    """Supported export formats."""

    CSV = "csv"
    GEOJSON = "geojson"
    SHAPEFILE = "shapefile"
    NETCDF = "netcdf"
    GEOTIFF = "geotiff"
    JSON = "json"


class AggregationMethod(str, Enum):
    """Supported aggregation methods."""

    MEAN = "mean"
    MAX = "max"
    MIN = "min"
    SUM = "sum"


def ensure_path(path: str | Path) -> Path:
    """Ensure path exists and return Path object."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


class BaseExporter:
    """Base class for data exporters."""

    def __init__(self, dataset: xr.Dataset, pollution_variable: str):
        """Initialize exporter with dataset and variable."""
        self.dataset = dataset
        self.pollution_variable = pollution_variable

    def _get_data(self) -> xr.DataArray:
        """Get data array for variable."""
        return self.dataset[self.pollution_variable]

    def _aggregate_data(
        self, data: xr.DataArray, method: str, dim: str = "time"
    ) -> xr.DataArray:
        """Aggregate data along dimension using method."""
        method = AggregationMethod(method)
        if method == AggregationMethod.MEAN:
            return data.mean(dim=dim)
        if method == AggregationMethod.MAX:
            return data.max(dim=dim)
        if method == AggregationMethod.MIN:
            return data.min(dim=dim)
        if method == AggregationMethod.SUM:
            return data.sum(dim=dim)
        raise ValueError(f"Unsupported aggregation method: {method}")

    def _select_time(
        self,
        data: xr.DataArray,
        time_index: int | str | slice,
        aggregation_method: str | None = None,
    ) -> xr.DataArray:
        """Select and potentially aggregate time data."""
        if isinstance(time_index, slice):
            data = data.sel(time=time_index)
            if aggregation_method:
                data = self._aggregate_data(data, aggregation_method)
            else:
                data = data.isel(time=0)
        elif isinstance(time_index, str):
            data = data.sel(time=time_index, method="nearest")
        else:
            data = data.isel(time=time_index)
        return data
