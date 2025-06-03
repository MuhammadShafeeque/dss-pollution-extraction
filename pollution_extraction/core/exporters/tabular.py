"""Tabular data export functionality."""

from pathlib import Path
from typing import Literal

import pandas as pd
import xarray as xr

from . import BaseExporter, ExportFormat, ensure_path, logger

FormatList = list[Literal["csv", "json"]]


class TabularExporter(BaseExporter):
    """Export data to tabular formats like CSV and JSON."""

    def to_csv(
        self,
        output_path: str | Path,
        include_coordinates: bool = True,
        time_format: str = "%Y-%m-%d",
        spatial_aggregation: str | None = None,
    ) -> None:
        """Export data to CSV format.

        Parameters
        ----------
        output_path : str or Path
            Output file path
        include_coordinates : bool
            Whether to include x, y coordinates
        time_format : str
            Time format string
        spatial_aggregation : str, optional
            Spatial aggregation method ('mean', 'max', 'min', 'sum')
        """
        output_path = ensure_path(output_path)
        data = self._get_data()

        if spatial_aggregation:
            data = self._aggregate_data(data, spatial_aggregation, dim=["x", "y"])
            df = data.to_dataframe().reset_index()
        else:
            df = data.to_dataframe().reset_index()

        # Format time column
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"]).dt.strftime(time_format)

        # Remove coordinate columns if not needed
        if not include_coordinates and spatial_aggregation:
            coord_cols = ["x", "y"]
            df = df.drop(columns=[col for col in coord_cols if col in df.columns])

        df.to_csv(output_path, index=False)
        logger.info(f"Data exported to CSV: {output_path}")

    def time_series_to_formats(
        self,
        time_series_data: xr.Dataset,
        output_dir: str | Path,
        formats: FormatList | None = None,
        base_filename: str = "time_series",
    ) -> dict[str, Path]:
        """Export time series data to multiple formats.

        Parameters
        ----------
        time_series_data : xr.Dataset
            Time series data
        output_dir : str or Path
            Output directory
        formats : list of str
            List of output formats ('csv', 'json')
        base_filename : str
            Base filename for outputs

        Returns:
        -------
        dict
            Dictionary mapping format names to output file paths
        """
        if formats is None:
            formats = ["csv"]
        output_dir = ensure_path(output_dir)
        output_paths = {}

        df = time_series_data.to_dataframe().reset_index()

        if ExportFormat.CSV.value in formats:
            csv_path = output_dir / f"{base_filename}.csv"
            df.to_csv(csv_path, index=False)
            output_paths["csv"] = csv_path

        if ExportFormat.JSON.value in formats:
            json_path = output_dir / f"{base_filename}.json"
            df.to_json(json_path, orient="records", date_format="iso")
            output_paths["json"] = json_path

        logger.info(f"Time series data exported to: {list(output_paths.values())}")
        return output_paths
