"""
Spatial extraction module for pollution data.
"""

import xarray as xr
import pandas as pd
import geopandas as gpd
import numpy as np
import rioxarray as rxr
from pathlib import Path
from typing import Union, List, Dict, Optional, Tuple
import logging
from shapely.geometry import Point, Polygon
import json

logger = logging.getLogger(__name__)


class SpatialExtractor:
    """
    Class for spatial extraction of pollution data.

    Supports extraction using various spatial formats including shapefiles,
    GeoJSON, CSV with coordinates, and specific locations.
    """

    def __init__(self, dataset: xr.Dataset, pollution_variable: str):
        """
        Initialize the spatial extractor.

        Parameters
        ----------
        dataset : xr.Dataset
            The pollution dataset with spatial coordinates
        pollution_variable : str
            Name of the pollution variable to extract
        """
        self.dataset = dataset
        self.pollution_variable = pollution_variable
        self._ensure_spatial_coords()

    def _ensure_spatial_coords(self) -> None:
        """Ensure the dataset has proper spatial coordinates."""
        if "x" not in self.dataset.coords or "y" not in self.dataset.coords:
            raise ValueError("Dataset must have 'x' and 'y' coordinates")

        # Set spatial dims for rioxarray if CRS info available
        if "crs" in self.dataset:
            try:
                crs_wkt = self.dataset.crs.attrs.get("crs_wkt", "")
                if crs_wkt:
                    self.dataset = self.dataset.rio.write_crs(crs_wkt)
                    self.dataset = self.dataset.rio.set_spatial_dims(
                        x_dim="x", y_dim="y"
                    )
            except Exception as e:
                logger.warning(f"Could not set CRS information: {e}")

    def load_spatial_boundaries(
        self, file_path: Union[str, Path], format_type: str = "auto"
    ) -> gpd.GeoDataFrame:
        """
        Load spatial boundaries from various file formats.

        Parameters
        ----------
        file_path : str or Path
            Path to the spatial boundary file
        format_type : str
            Format type ('shapefile', 'geojson', 'csv', 'auto')

        Returns
        -------
        gpd.GeoDataFrame
            Loaded spatial boundaries
        """
        file_path = Path(file_path)

        if format_type == "auto":
            format_type = self._detect_format(file_path)

        if format_type == "shapefile":
            gdf = gpd.read_file(file_path)
        elif format_type == "geojson":
            gdf = gpd.read_file(file_path)
        elif format_type == "csv":
            gdf = self._load_csv_with_coordinates(file_path)
        elif format_type == "json":
            gdf = self._load_json_coordinates(file_path)
        else:
            raise ValueError(f"Unsupported format: {format_type}")

        return gdf

    def _detect_format(self, file_path: Path) -> str:
        """Auto-detect file format based on extension."""
        suffix = file_path.suffix.lower()

        format_map = {
            ".shp": "shapefile",
            ".geojson": "geojson",
            ".json": "json",
            ".csv": "csv",
        }

        return format_map.get(suffix, "unknown")

    def _load_csv_with_coordinates(self, file_path: Path) -> gpd.GeoDataFrame:
        """
        Load CSV file with coordinate columns.

        Expected columns: 'lon', 'lat' or 'longitude', 'latitude' or 'x', 'y'
        """
        df = pd.read_csv(file_path)

        # Try to find coordinate columns
        lon_cols = ["lon", "longitude", "x", "long"]
        lat_cols = ["lat", "latitude", "y"]

        lon_col = next((col for col in lon_cols if col in df.columns), None)
        lat_col = next((col for col in lat_cols if col in df.columns), None)

        if not lon_col or not lat_col:
            raise ValueError(
                f"Could not find coordinate columns in CSV. "
                f"Available columns: {df.columns.tolist()}"
            )

        # Create geometry points
        geometry = [Point(lon, lat) for lon, lat in zip(df[lon_col], df[lat_col])]
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

        return gdf

    def _load_json_coordinates(self, file_path: Path) -> gpd.GeoDataFrame:
        """Load JSON file with coordinate information."""
        with open(file_path, "r") as f:
            data = json.load(f)

        # Handle different JSON structures
        if isinstance(data, list):
            # List of coordinate objects
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            if "features" in data:
                # GeoJSON-like structure
                return gpd.read_file(file_path)
            else:
                # Single object or dict of objects
                df = pd.DataFrame(
                    [data]
                    if not isinstance(list(data.values())[0], dict)
                    else list(data.values())
                )

        # Try to create geometries
        if "lon" in df.columns and "lat" in df.columns:
            geometry = [Point(lon, lat) for lon, lat in zip(df["lon"], df["lat"])]
        elif "longitude" in df.columns and "latitude" in df.columns:
            geometry = [
                Point(lon, lat) for lon, lat in zip(df["longitude"], df["latitude"])
            ]
        else:
            raise ValueError("Could not find coordinate columns in JSON")

        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
        return gdf

    def extract_points(
        self,
        locations: Union[gpd.GeoDataFrame, List[Tuple[float, float]], str, Path],
        method: str = "nearest",
        buffer_distance: Optional[float] = None,
    ) -> xr.Dataset:
        """
        Extract data at specific point locations.

        Parameters
        ----------
        locations : gpd.GeoDataFrame, list of tuples, str, or Path
            Point locations for extraction
        method : str
            Extraction method ('nearest', 'linear', 'cubic')
        buffer_distance : float, optional
            Buffer distance around points for aggregation

        Returns
        -------
        xr.Dataset
            Extracted data at point locations
        """        # Handle different input types
        if isinstance(locations, (str, Path)):
            gdf = self.load_spatial_boundaries(locations)
        elif isinstance(locations, list):
            # List of (x, y) tuples - assume they are in the same CRS as the dataset
            # Get the dataset's CRS or use a default
            dataset_crs = getattr(self.dataset.rio, 'crs', None)
            if dataset_crs is None:
                # Assume LAEA projection for European data if no CRS info
                dataset_crs = "EPSG:3035"
                logger.warning(f"No CRS found in dataset, assuming {dataset_crs}")
            
            gdf = gpd.GeoDataFrame(
                {"id": range(len(locations))},
                geometry=[Point(x, y) for x, y in locations],
                crs=dataset_crs,
            )
        else:
            gdf = locations.copy()

        # Ensure proper CRS - only transform if CRS are different and both are defined
        if hasattr(self.dataset, 'rio') and self.dataset.rio.crs is not None:
            if gdf.crs != self.dataset.rio.crs:
                gdf = gdf.to_crs(self.dataset.rio.crs)
        else:
            logger.warning("Dataset has no CRS information, assuming coordinates are in correct projection")

        results = []

        for idx, row in gdf.iterrows():
            point = row.geometry

            if buffer_distance:
                # Create buffer and aggregate
                buffered = point.buffer(buffer_distance)
                extracted = self._extract_polygon(buffered, method="mean")
            else:                # Extract at exact point
                x_coord = point.x
                y_coord = point.y

                # Validate coordinates are within dataset bounds
                x_min, x_max = float(self.dataset.x.min()), float(self.dataset.x.max())
                y_min, y_max = float(self.dataset.y.min()), float(self.dataset.y.max())
                
                if not (x_min <= x_coord <= x_max and y_min <= y_coord <= y_max):
                    logger.warning(f"Point ({x_coord}, {y_coord}) is outside dataset bounds: "
                                   f"x=[{x_min}, {x_max}], y=[{y_min}, {y_max}]")
                    # Create NaN result with same structure
                    extracted = self.dataset.isel(x=0, y=0) * np.nan
                else:
                    try:
                        extracted = self.dataset.sel(x=x_coord, y=y_coord, method=method)
                    except (KeyError, ValueError) as e:
                        logger.error(f"Failed to extract at point ({x_coord}, {y_coord}): {e}")
                        # Create NaN result with same structure
                        extracted = self.dataset.isel(x=0, y=0) * np.nan

            # Add location identifier
            extracted.coords["location_id"] = idx

            # Add metadata from original GeoDataFrame
            for col in gdf.columns:
                if col != "geometry":
                    extracted.coords[f"location_{col}"] = row[col]

            results.append(extracted)

        # Combine results
        combined = xr.concat(results, dim="location_id")
        return combined

    def extract_polygons(
        self,
        polygons: Union[gpd.GeoDataFrame, str, Path],
        aggregation_method: str = "mean",
        mask_and_scale: bool = True,
    ) -> xr.Dataset:
        """
        Extract data within polygon boundaries.

        Parameters
        ----------
        polygons : gpd.GeoDataFrame, str, or Path
            Polygon boundaries for extraction
        aggregation_method : str
            Method for aggregating values within polygons
        mask_and_scale : bool
            Whether to mask areas outside polygons

        Returns
        -------
        xr.Dataset
            Extracted data for each polygon
        """
        # Handle different input types
        if isinstance(polygons, (str, Path)):
            gdf = self.load_spatial_boundaries(polygons)
        else:
            gdf = polygons.copy()

        # Ensure proper CRS
        if gdf.crs != self.dataset.rio.crs:
            gdf = gdf.to_crs(self.dataset.rio.crs)

        results = []

        for idx, row in gdf.iterrows():
            polygon = row.geometry

            # Extract data for this polygon
            extracted = self._extract_polygon(
                polygon, method=aggregation_method, mask_and_scale=mask_and_scale
            )

            # Add polygon identifier
            extracted.coords["polygon_id"] = idx

            # Add metadata from original GeoDataFrame
            for col in gdf.columns:
                if col != "geometry":
                    extracted.coords[f"polygon_{col}"] = row[col]

            results.append(extracted)

        # Combine results
        combined = xr.concat(results, dim="polygon_id")
        return combined

    def _extract_polygon(
        self, polygon: Polygon, method: str = "mean", mask_and_scale: bool = True
    ) -> xr.Dataset:
        """Extract data within a single polygon."""
        try:
            # Clip dataset to polygon bounds
            minx, miny, maxx, maxy = polygon.bounds
            clipped = self.dataset.sel(x=slice(minx, maxx), y=slice(miny, maxy))

            if mask_and_scale:
                # Create mask for the polygon
                clipped_with_mask = clipped.rio.clip([polygon], drop=True)

                if method == "mean":
                    result = clipped_with_mask.mean(dim=["x", "y"])
                elif method == "sum":
                    result = clipped_with_mask.sum(dim=["x", "y"])
                elif method == "min":
                    result = clipped_with_mask.min(dim=["x", "y"])
                elif method == "max":
                    result = clipped_with_mask.max(dim=["x", "y"])
                elif method == "std":
                    result = clipped_with_mask.std(dim=["x", "y"])
                else:
                    raise ValueError(f"Unsupported aggregation method: {method}")
            else:
                # Simple spatial aggregation without masking
                if method == "mean":
                    result = clipped.mean(dim=["x", "y"])
                elif method == "sum":
                    result = clipped.sum(dim=["x", "y"])
                elif method == "min":
                    result = clipped.min(dim=["x", "y"])
                elif method == "max":
                    result = clipped.max(dim=["x", "y"])
                elif method == "std":
                    result = clipped.std(dim=["x", "y"])
                else:
                    raise ValueError(f"Unsupported aggregation method: {method}")

            return result

        except Exception as e:
            logger.error(f"Error extracting polygon data: {e}")
            # Return NaN dataset with same structure
            result = self.dataset.isel(x=0, y=0) * np.nan
            return result

    def extract_nuts3_regions(
        self, nuts3_file: Union[str, Path], aggregation_method: str = "mean"
    ) -> xr.Dataset:
        """
        Extract data for NUTS3 regions in Europe.

        Parameters
        ----------
        nuts3_file : str or Path
            Path to NUTS3 boundary file
        aggregation_method : str
            Method for aggregating values within regions

        Returns
        -------
        xr.Dataset
            Extracted data for each NUTS3 region
        """
        # Load NUTS3 boundaries
        nuts3_gdf = self.load_spatial_boundaries(nuts3_file)

        # Ensure NUTS3 identifier column exists
        nuts3_col = None
        for col in ["NUTS_ID", "nuts_id", "NUTS3", "nuts3", "id", "ID"]:
            if col in nuts3_gdf.columns:
                nuts3_col = col
                break

        if not nuts3_col:
            logger.warning("No NUTS3 identifier column found. Using index.")
            nuts3_gdf["nuts3_id"] = nuts3_gdf.index
            nuts3_col = "nuts3_id"

        # Extract data for each NUTS3 region
        return self.extract_polygons(nuts3_gdf, aggregation_method)

    def extract_with_dates(
        self,
        locations: Union[gpd.GeoDataFrame, str, Path],
        date_column: str = "date",
        days_before: int = 0,
        aggregation_method: str = "mean",
    ) -> xr.Dataset:
        """
        Extract data at locations for specific dates (with optional days before).

        Parameters
        ----------
        locations : gpd.GeoDataFrame, str, or Path
            Locations with associated dates
        date_column : str
            Column name containing dates
        days_before : int
            Number of days before each date to include
        aggregation_method : str
            Method for temporal aggregation if days_before > 0

        Returns
        -------
        xr.Dataset
            Extracted data for each location and date
        """
        # Load locations
        if isinstance(locations, (str, Path)):
            gdf = self.load_spatial_boundaries(locations)
        else:
            gdf = locations.copy()

        if date_column not in gdf.columns:
            raise ValueError(f"Date column '{date_column}' not found in locations data")

        # Ensure proper CRS
        if gdf.crs != self.dataset.rio.crs:
            gdf = gdf.to_crs(self.dataset.rio.crs)

        results = []

        for idx, row in gdf.iterrows():
            point = row.geometry
            event_date = pd.to_datetime(row[date_column])

            # Calculate time range
            if days_before > 0:
                start_date = event_date - pd.Timedelta(days=days_before)
                time_slice = slice(start_date, event_date)
            else:
                time_slice = event_date

            # Extract spatial data
            x_coord = point.x
            y_coord = point.y

            extracted = self.dataset.sel(
                x=x_coord, y=y_coord, time=time_slice, method="nearest"
            )

            # Apply temporal aggregation if needed
            if days_before > 0:
                if aggregation_method == "mean":
                    extracted = extracted.mean(dim="time")
                elif aggregation_method == "sum":
                    extracted = extracted.sum(dim="time")
                elif aggregation_method == "max":
                    extracted = extracted.max(dim="time")
                elif aggregation_method == "min":
                    extracted = extracted.min(dim="time")

            # Add metadata
            extracted.coords["location_id"] = idx
            extracted.coords["event_date"] = event_date
            extracted.coords["days_before"] = days_before

            # Add other attributes from GeoDataFrame
            for col in gdf.columns:
                if col not in ["geometry", date_column]:
                    extracted.coords[f"location_{col}"] = row[col]

            results.append(extracted)

        # Combine results
        combined = xr.concat(results, dim="location_id")
        return combined

    def spatial_subset(self, bounds: Dict[str, float]) -> xr.Dataset:
        """
        Extract data for a bounding box.

        Parameters
        ----------
        bounds : dict
            Bounding box with keys 'minx', 'miny', 'maxx', 'maxy'

        Returns
        -------
        xr.Dataset
            Spatially subset dataset
        """
        return self.dataset.sel(
            x=slice(bounds["minx"], bounds["maxx"]),
            y=slice(bounds["miny"], bounds["maxy"]),
        )
