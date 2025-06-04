"""Common type definitions for exporters."""

from typing import Literal

# Export format types
TabularFormatList = list[Literal["csv", "json"]]
SpatialFormatList = list[Literal["geojson", "shapefile", "csv"]]
RasterFormatList = list[Literal["geotiff", "netcdf"]]
