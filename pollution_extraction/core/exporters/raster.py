"""Raster data export functionality."""

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from rasterio.crs import CRS
from rasterio.transform import from_bounds

from ..logging_utils import logger
from .base import BaseExporter, ensure_path


class RasterExporter(BaseExporter):
    """Export data to raster formats like GeoTIFF."""

    def _get_crs(self, default: str = "EPSG:3035") -> CRS:
        """Get CRS from dataset or return default."""
        if "crs" in self.dataset:
            crs_wkt = self.dataset.crs.attrs.get("crs_wkt", "")
            if crs_wkt:
                try:
                    return CRS.from_wkt(crs_wkt)
                except Exception as e:
                    logger.warning(
                        f"Could not parse CRS from WKT ({e!s}), using {default}"
                    )
        return CRS.from_string(default)

    def _get_transform(self, x_coords: np.ndarray, y_coords: np.ndarray) -> tuple:
        """Calculate transform for GeoTIFF."""
        x_res = x_coords[1] - x_coords[0]
        y_res = y_coords[1] - y_coords[0]
        return from_bounds(
            x_coords[0] - x_res / 2,
            y_coords[0] - y_res / 2,
            x_coords[-1] + x_res / 2,
            y_coords[-1] + y_res / 2,
            len(x_coords),
            len(y_coords),
        )

    def _prepare_array(
        self, data: xr.DataArray, nodata_value: float, dtype: str
    ) -> np.ndarray:
        """Prepare array for GeoTIFF export."""
        data_array = data.values
        data_array = np.where(np.isnan(data_array), nodata_value, data_array)

        # Flip y-axis if needed (rasterio expects top-left origin)
        if data.y.values[0] < data.y.values[-1]:
            data_array = np.flipud(data_array)

        return data_array.astype(dtype)

    def _prepare_metadata(
        self, data: xr.DataArray, aggregation_method: str | None = None
    ) -> dict:
        """Prepare metadata for GeoTIFF."""
        metadata = {
            "pollution_variable": self.pollution_variable,
            "source_file": str(getattr(self.dataset, "source", "")),
            "creation_date": pd.Timestamp.now().isoformat(),
        }

        # Add time information
        if hasattr(data, "time"):
            if data.time.size == 1:
                time_str = pd.to_datetime(data.time.values).isoformat()
                metadata["time"] = time_str
            else:
                start = pd.to_datetime(data.time.values[0]).isoformat()
                end = pd.to_datetime(data.time.values[-1]).isoformat()
                metadata["time_range"] = f"{start} to {end}"
                if aggregation_method:
                    metadata["temporal_aggregation"] = aggregation_method

        # Add variable attributes
        if hasattr(data, "attrs"):
            for key, value in data.attrs.items():
                if isinstance(value, str | int | float):
                    metadata[key] = str(value)

        return metadata

    def to_geotiff(
        self,
        output_path: str | Path,
        time_index: int | str | slice = 0,
        aggregation_method: str | None = None,
        nodata_value: float = -9999.0,
        compress: str = "lzw",
        dtype: str = "float32",
    ) -> None:
        """Export data to GeoTIFF format.

        Parameters
        ----------
        output_path : str or Path
            Output file path
        time_index : int, str, or slice
            Time index/slice to export
        aggregation_method : str, optional
            Temporal aggregation method if time_index is a slice
        nodata_value : float
            NoData value for the GeoTIFF
        compress : str
            Compression method
        dtype : str
            Data type for the output

        """
        output_path = ensure_path(output_path)

        # Process data
        data = self._get_data()
        data = self._select_time(data, time_index, aggregation_method)

        # Get spatial information
        transform = self._get_transform(data.x.values, data.y.values)
        crs = self._get_crs()

        # Prepare arrays and metadata
        data_array = self._prepare_array(data, nodata_value, dtype)
        metadata = self._prepare_metadata(data, aggregation_method)

        # Write GeoTIFF
        with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            height=data_array.shape[0],
            width=data_array.shape[1],
            count=1,
            dtype=dtype,
            crs=crs,
            transform=transform,
            nodata=nodata_value,
            compress=compress,
            tiled=True,
            blockxsize=256,
            blockysize=256,
        ) as dst:
            dst.write(data_array, 1)
            dst.update_tags(**metadata)

        logger.info(f"Data exported to GeoTIFF: {output_path}")
