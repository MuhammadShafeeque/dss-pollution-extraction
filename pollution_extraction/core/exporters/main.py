"""Main data exporter combining specialized exporters."""

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .base import logger
from .raster import RasterExporter
from .spatial import SpatialExporter
from .tabular import TabularExporter

__all__ = ["DataExporter"]


class DataExporter(RasterExporter, SpatialExporter, TabularExporter):
    """Export pollution data to various formats."""

    def create_metadata_file(
        self, output_path: str | Path, processing_info: dict[str, Any] | None = None
    ) -> None:
        """Create a metadata file for the exported data.

        Parameters
        ----------
        output_path : str or Path
            Output path for metadata file
        processing_info : dict, optional
            Additional processing information
        """
        output_path = Path(output_path)

        metadata = {
            "dataset_info": {
                "pollution_variable": self.pollution_variable,
                "dimensions": dict(self.dataset.dims),
                "coordinates": list(self.dataset.coords),
                "time_range": {
                    "start": pd.to_datetime(self.dataset.time.values[0]).isoformat(),
                    "end": pd.to_datetime(self.dataset.time.values[-1]).isoformat(),
                },
                "spatial_bounds": {
                    "x_min": float(self.dataset.x.min()),
                    "x_max": float(self.dataset.x.max()),
                    "y_min": float(self.dataset.y.min()),
                    "y_max": float(self.dataset.y.max()),
                },
            },
            "variable_attributes": dict(self.dataset[self.pollution_variable].attrs),
            "global_attributes": dict(self.dataset.attrs),
            "export_info": {
                "export_date": pd.Timestamp.now().isoformat(),
                "software": "dss-pollution-extraction",
            },
        }

        if processing_info:
            metadata["processing_info"] = processing_info

        if "crs" in self.dataset:
            metadata["coordinate_system"] = dict(self.dataset.crs.attrs)

        with open(output_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Metadata file created: {output_path}")
