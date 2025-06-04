"""Spatial data export functionality."""

from pathlib import Path

import geopandas as gpd
import pandas as pd
import xarray as xr
from shapely.geometry import Point

from ..logging_utils import logger
from .base import (
    BaseExporter,
    ExportFormat,
    SpatialFormatList as FormatList,
    ensure_path,
)


class SpatialExporter(BaseExporter):
    """Export data to spatial formats like GeoJSON and Shapefile."""

    def _create_point_gdf(
        self,
        summary_df: pd.DataFrame,
        x_coords: list[float],
        y_coords: list[float],
        crs: str = "EPSG:3035",
    ) -> gpd.GeoDataFrame | None:
        """Create GeoDataFrame from point data."""
        if len(x_coords) != len(summary_df):
            return None

        geometry = [
            Point(x, y)
            for x, y in zip(x_coords, y_coords, strict=False)
            if x is not None and y is not None
        ]

        if not geometry:
            return None

        return gpd.GeoDataFrame(summary_df, geometry=geometry, crs=crs)

    def _extract_coordinates(self, df: pd.DataFrame) -> tuple[list[float], list[float]]:
        """Extract x,y coordinates from DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with point location data

        Returns
        -------
        tuple[list[float], list[float]]
            Lists of x and y coordinates

        """
        x_coords: list[float] = []
        y_coords: list[float] = []

        for idx in df["location_id"].unique():
            location_data = df[df["location_id"] == idx]
            x_coord = location_data.get("location_x", None)
            y_coord = location_data.get("location_y", None)

            if x_coord is None or y_coord is None:
                if all(col in location_data.columns for col in ["x", "y"]):
                    x_coord = location_data["x"].iloc[0]
                    y_coord = location_data["y"].iloc[0]

            x_coords.append(x_coord if x_coord is not None else 0.0)
            y_coords.append(y_coord if y_coord is not None else 0.0)

        return x_coords, y_coords

    def _extract_coordinates_from_location(
        self, location_data: pd.DataFrame
    ) -> tuple[float, float]:
        """Get coordinates from location data.

        Parameters
        ----------
        location_data : pd.DataFrame
            DataFrame containing location data

        Returns
        -------
        tuple[float, float]
            x and y coordinates

        """
        x_coord = location_data.get("location_x", None)
        y_coord = location_data.get("location_y", None)

        if x_coord is None or y_coord is None:
            cols = ["x", "y"]
            if all(col in location_data.columns for col in cols):
                x_coord = location_data["x"].iloc[0]
                y_coord = location_data["y"].iloc[0]

        return (
            x_coord if x_coord is not None else 0.0,
            y_coord if y_coord is not None else 0.0,
        )

    def _get_location_coordinates(
        self, df: pd.DataFrame
    ) -> tuple[list[float], list[float]]:
        """Get coordinates for all locations.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with location data

        Returns
        -------
        tuple[list[float], list[float]]
            Lists of x and y coordinates

        """
        coords: list[tuple[float, float]] = [
            self._extract_coordinates_from_location(df[df["location_id"] == idx])
            for idx in df["location_id"].unique()
        ]
        return [c[0] for c in coords], [c[1] for c in coords]

    def _create_stat_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create statistical summary for points/polygons.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with values

        Returns
        -------
        pd.DataFrame
            Summary statistics

        """
        stats = ["mean", "min", "max", "std", "count"]
        summary = (
            df.groupby("location_id").agg({self.pollution_variable: stats}).round(4)
        )
        summary.columns = ["_".join(col).strip() for col in summary.columns.values]
        return summary.reset_index()

    def _export_csv_files(
        self, main_df: pd.DataFrame, summary_df: pd.DataFrame, out_dir: Path, name: str
    ) -> dict[str, Path]:
        """Export main and summary data to CSV.

        Parameters
        ----------
        main_df : pd.DataFrame
            Main DataFrame to export
        summary_df : pd.DataFrame
            Summary DataFrame to export
        out_dir : Path
            Output directory
        name : str
            Base name for files

        Returns
        -------
        dict[str, Path]
            Mapping of output types to file paths

        """
        paths: dict[str, Path] = {}

        # Export main data
        main_path = out_dir / f"{name}.csv"
        main_df.to_csv(main_path, index=False)
        paths["csv"] = main_path

        # Export summary
        summary_path = out_dir / f"{name}_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        paths["csv_summary"] = summary_path

        return paths

    def _create_summary_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create summary DataFrame with statistics."""
        summary_df = (
            df.groupby("location_id")
            .agg({self.pollution_variable: ["mean", "min", "max", "std", "count"]})
            .round(4)
        )
        summary_df.columns = [
            "_".join(col).strip() for col in summary_df.columns.values
        ]
        return summary_df.reset_index()

    def _export_to_csv(
        self,
        df: pd.DataFrame,
        summary_df: pd.DataFrame,
        output_dir: Path,
        base_filename: str,
    ) -> dict[str, Path]:
        """Export data to CSV format."""
        paths = {}
        csv_path = output_dir / f"{base_filename}.csv"
        df.to_csv(csv_path, index=False)
        paths["csv"] = csv_path

        summary_csv_path = output_dir / f"{base_filename}_summary.csv"
        summary_df.to_csv(summary_csv_path, index=False)
        paths["csv_summary"] = summary_csv_path
        return paths

    def extracted_points_to_formats(
        self,
        extracted_data: xr.Dataset,
        output_dir: str | Path,
        formats: FormatList | None = None,
        base_filename: str = "extracted_points",
    ) -> dict[str, Path]:
        """Export extracted point data to multiple formats."""
        if formats is None:
            formats = ["csv", "geojson"]
        output_dir = ensure_path(output_dir)
        output_paths = {}

        # Convert to DataFrame
        df = extracted_data.to_dataframe().reset_index()

        # Get coordinate information
        x_coords, y_coords = self._extract_coordinates(df)

        # Create summary DataFrame
        summary_df = self._create_summary_df(df)

        # Add coordinates if available
        if len(x_coords) == len(summary_df):
            summary_df["x"] = x_coords
            summary_df["y"] = y_coords

        # Export to different formats
        if ExportFormat.CSV.value in formats:
            csv_path = output_dir / f"{base_filename}.csv"
            df.to_csv(csv_path, index=False)
            output_paths["csv"] = csv_path

            summary_csv_path = output_dir / f"{base_filename}_summary.csv"
            summary_df.to_csv(summary_csv_path, index=False)
            output_paths["csv_summary"] = summary_csv_path

        # Create GeoDataFrame for spatial formats if coordinates available
        gdf = self._create_point_gdf(summary_df, x_coords, y_coords)

        if gdf is not None:
            if ExportFormat.GEOJSON.value in formats:
                geojson_path = output_dir / f"{base_filename}.geojson"
                gdf.to_file(geojson_path, driver="GeoJSON")
                output_paths["geojson"] = geojson_path

            if ExportFormat.SHAPEFILE.value in formats:
                shp_path = output_dir / f"{base_filename}.shp"
                gdf.to_file(shp_path, driver="ESRI Shapefile")
                output_paths["shapefile"] = shp_path

        logger.info(f"Extracted point data exported to: {list(output_paths.values())}")
        return output_paths

    def extracted_polygons_to_formats(
        self,
        extracted_data: xr.Dataset,
        original_polygons: gpd.GeoDataFrame,
        output_dir: str | Path,
        formats: FormatList | None = None,
        base_filename: str = "extracted_polygons",
    ) -> dict[str, Path]:
        """Export extracted polygon data to multiple formats."""
        if formats is None:
            formats = ["csv", "geojson"]
        output_dir = ensure_path(output_dir)
        output_paths = {}

        # Convert to DataFrame
        df = extracted_data.to_dataframe().reset_index()

        # Create summary DataFrame
        summary_df = (
            df.groupby("polygon_id")
            .agg({self.pollution_variable: ["mean", "min", "max", "std", "count"]})
            .round(4)
        )
        summary_df.columns = [
            "_".join(col).strip() for col in summary_df.columns.values
        ]
        summary_df = summary_df.reset_index()

        # Merge with original polygon attributes if possible
        if len(original_polygons) == len(summary_df):
            polygon_attrs = original_polygons.drop(columns=["geometry"]).reset_index(
                drop=True
            )
            summary_df = pd.concat([summary_df, polygon_attrs], axis=1)

            # Export to different formats
            if ExportFormat.CSV.value in formats:
                csv_path = output_dir / f"{base_filename}.csv"
                df.to_csv(csv_path, index=False)
                output_paths["csv"] = csv_path

                summary_csv_path = output_dir / f"{base_filename}_summary.csv"
                summary_df.to_csv(summary_csv_path, index=False)
                output_paths["csv_summary"] = summary_csv_path

            # Create spatial exports with original geometries
            gdf = gpd.GeoDataFrame(
                summary_df,
                geometry=original_polygons.geometry.values,
                crs=original_polygons.crs,
            )

            if ExportFormat.GEOJSON.value in formats:
                geojson_path = output_dir / f"{base_filename}.geojson"
                gdf.to_file(geojson_path, driver="GeoJSON")
                output_paths["geojson"] = geojson_path

            if ExportFormat.SHAPEFILE.value in formats:
                shp_path = output_dir / f"{base_filename}.shp"
                gdf.to_file(shp_path, driver="ESRI Shapefile")
                output_paths["shapefile"] = shp_path

        logger.info(
            f"Extracted polygon data exported to: {list(output_paths.values())}"
        )
        return output_paths
