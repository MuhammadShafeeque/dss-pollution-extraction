"""Core data reader module for pollution NetCDF files."""

import logging
from pathlib import Path

import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)


class PollutionDataReader:
    """Core class for reading and handling pollution data from NetCDF files.

    Supports various pollution types including BC, NO2, PM2.5, and PM10.
    """

    # Mapping of pollution types to their variable names and metadata
    POLLUTION_VARIABLES = {
        "bc": {
            "var_name": "BC_downscaled",
            "standard_name": "atmosphere_optical_thickness_due_to_black_carbon_ambient_aerosol_particles",
            "units": "10^-5 m",
            "description": "Black Carbon Aerosol Optical Depth",
        },
        "no2": {
            "var_name": "no2_downscaled",
            "standard_name": "mass_concentration_of_nitrogen_dioxide_in_air",
            "units": "μg/m³",
            "description": "Nitrogen Dioxide Concentration",
        },
        "pm25": {
            "var_name": "PM2p5_downscaled",
            "standard_name": "mass_concentration_of_pm2p5_ambient_aerosol_particles_in_air",
            "units": "μg/m³",
            "description": "PM2.5 Concentration",
        },
        "pm10": {
            "var_name": "PM10_downscaled",
            "standard_name": "mass_concentration_of_pm10_ambient_aerosol_particles_in_air",
            "units": "μg/m³",
            "description": "PM10 Concentration",
        },
    }

    def __init__(self, file_path: str | Path, pollution_type: str | None = None):
        """Initialize the pollution data reader.

        Parameters
        ----------
        file_path : str or Path
            Path to the NetCDF file
        pollution_type : str, optional
            Type of pollution data ('bc', 'no2', 'pm25', 'pm10')
            If None, will attempt to auto-detect
        """
        self.file_path = Path(file_path)
        self.pollution_type = pollution_type
        self.dataset = None
        self._load_dataset()

        if not self.pollution_type:
            self.pollution_type = self._detect_pollution_type()

        self.variable_info = self.POLLUTION_VARIABLES.get(self.pollution_type, {})

    def _load_dataset(self) -> None:
        """Load the NetCDF dataset using xarray."""
        try:
            self.dataset = xr.open_dataset(self.file_path, chunks="auto")
            logger.info(f"Successfully loaded dataset from {self.file_path}")
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise

    def _detect_pollution_type(self) -> str:
        """Auto-detect pollution type based on variable names in the dataset.

        Returns:
        -------
        str
            Detected pollution type
        """
        variables = list(self.dataset.data_vars)

        for poll_type, info in self.POLLUTION_VARIABLES.items():
            if info["var_name"] in variables:
                logger.info(f"Auto-detected pollution type: {poll_type}")
                return poll_type

        raise ValueError(f"Could not detect pollution type from variables: {variables}")

    @property
    def data_variable(self) -> xr.DataArray:
        """Get the main pollution data variable."""
        var_name = self.variable_info["var_name"]
        return self.dataset[var_name]

    @property
    def time_range(self) -> tuple:
        """Get the time range of the dataset."""
        time_values = self.dataset.time.values
        return pd.to_datetime(time_values[0]), pd.to_datetime(time_values[-1])

    @property
    def spatial_bounds(self) -> dict[str, float]:
        """Get spatial bounds of the dataset."""
        return {
            "x_min": float(self.dataset.x.min()),
            "x_max": float(self.dataset.x.max()),
            "y_min": float(self.dataset.y.min()),
            "y_max": float(self.dataset.y.max()),
        }

    @property
    def coordinate_system(self) -> dict:
        """Get coordinate system information."""
        crs_info = {}
        if "crs" in self.dataset:
            crs_attrs = self.dataset.crs.attrs
            crs_info = {
                "grid_mapping_name": crs_attrs.get("grid_mapping_name", ""),
                "longitude_of_projection_origin": crs_attrs.get(
                    "longitude_of_projection_origin", 0
                ),
                "latitude_of_projection_origin": crs_attrs.get(
                    "latitude_of_projection_origin", 0
                ),
                "false_easting": crs_attrs.get("false_easting", 0),
                "false_northing": crs_attrs.get("false_northing", 0),
                "crs_wkt": crs_attrs.get("crs_wkt", ""),
            }
        return crs_info

    def get_basic_info(self) -> dict:
        """Get basic information about the dataset.

        Returns:
        -------
        dict
            Basic dataset information
        """
        start_time, end_time = self.time_range

        return {
            "file_path": str(self.file_path),
            "pollution_type": self.pollution_type,
            "variable_name": self.variable_info.get("var_name", ""),
            "units": self.variable_info.get("units", ""),
            "description": self.variable_info.get("description", ""),
            "time_range": (start_time, end_time),
            "total_time_steps": len(self.dataset.time),
            "spatial_dimensions": {"x": len(self.dataset.x), "y": len(self.dataset.y)},
            "spatial_bounds": self.spatial_bounds,
            "coordinate_system": self.coordinate_system,
        }

    def get_data_summary(self) -> dict:
        """Get statistical summary of the pollution data.

        Returns:
        -------
        dict
            Statistical summary
        """
        data = self.data_variable

        return {
            "min": float(data.min()),
            "max": float(data.max()),
            "mean": float(data.mean()),
            "std": float(data.std()),
            "median": float(data.median()),
            "count_valid": int((~data.isnull()).sum()),
            "count_total": int(data.size),
            "missing_percentage": float(data.isnull().sum() / data.size * 100),
        }

    def select_time_range(self, start_date: str, end_date: str) -> xr.Dataset:
        """Select data for a specific time range.

        Parameters
        ----------
        start_date : str
            Start date in format 'YYYY-MM-DD'
        end_date : str
            End date in format 'YYYY-MM-DD'

        Returns:
        -------
        xr.Dataset
            Dataset subset for the time range
        """
        return self.dataset.sel(time=slice(start_date, end_date))

    def select_time_points(self, dates: list[str]) -> xr.Dataset:
        """Select data for specific time points.

        Parameters
        ----------
        dates : list of str
            List of dates in format 'YYYY-MM-DD'

        Returns:
        -------
        xr.Dataset
            Dataset subset for the specified dates
        """
        date_objects = pd.to_datetime(dates)
        return self.dataset.sel(time=date_objects, method="nearest")

    def close(self) -> None:
        """Close the dataset."""
        if self.dataset:
            self.dataset.close()
            logger.info("Dataset closed")
