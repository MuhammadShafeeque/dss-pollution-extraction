"""Main pollution analyzer class that coordinates all components."""

import logging
from pathlib import Path
from typing import Any

import geopandas as gpd
import xarray as xr
from matplotlib.figure import Figure

from .core import DataExporter
from .core.data_reader import PollutionDataReader
from .core.data_visualizer import DataVisualizer
from .core.spatial_extractor import SpatialExtractor
from .core.temporal_aggregator import TemporalAggregator

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PollutionAnalyzer:
    """Main class for pollution data analysis.

    Provides a high-level interface for reading, processing, analyzing,
    visualizing, and exporting pollution data from NetCDF files.
    """

    def __init__(self, file_path: str | Path, pollution_type: str | None = None):
        """Initialize the pollution analyzer.

        Parameters
        ----------
        file_path : str or Path
            Path to the NetCDF file
        pollution_type : str, optional
            Type of pollution data ('bc', 'no2', 'pm25', 'pm10')

        """
        self.file_path = Path(file_path)
        self.pollution_type = pollution_type

        # Initialize data reader
        self.reader = PollutionDataReader(file_path, pollution_type)
        self.dataset = self.reader.dataset
        self.pollution_variable = self.reader.variable_info["var_name"]

        # Initialize other components
        self.temporal_aggregator = TemporalAggregator(
            self.dataset, self.pollution_variable
        )
        self.spatial_extractor = SpatialExtractor(self.dataset, self.pollution_variable)
        self.visualizer = DataVisualizer(
            self.dataset, self.pollution_variable, self.pollution_type
        )
        self.exporter = DataExporter(self.dataset, self.pollution_variable)

        logger.info(f"PollutionAnalyzer initialized for {self.pollution_type} data")

    def get_info(self) -> dict:
        """Get comprehensive information about the dataset.

        Returns
        -------
        dict
            Dataset information including basic info and statistics

        """
        basic_info = self.reader.get_basic_info()
        data_summary = self.reader.get_data_summary()

        return {"basic_info": basic_info, "data_summary": data_summary}

    def print_summary(self) -> None:
        """Print a formatted summary of the dataset."""
        info = self.get_info()
        basic = info["basic_info"]
        summary = info["data_summary"]

        print("\n" + "=" * 50)
        print("POLLUTION DATASET SUMMARY")
        print("=" * 50)
        print(f"File: {basic['file_path']}")
        print(f"Pollution Type: {basic['pollution_type'].upper()}")
        print(f"Variable: {basic['variable_name']}")
        print(f"Units: {basic['units']}")
        print(f"Description: {basic['description']}")

        print("\nTEMPORAL INFORMATION:")
        print(f"Time Range: {basic['time_range'][0]} to {basic['time_range'][1]}")
        print(f"Total Time Steps: {basic['total_time_steps']}")

        print("\nSPATIAL INFORMATION:")
        print(
            f"Dimensions: {basic['spatial_dimensions']['x']} x "
            f"{basic['spatial_dimensions']['y']}"
        )
        print(
            f"X Range: {basic['spatial_bounds']['x_min']:.0f} to {basic['spatial_bounds']['x_max']:.0f}"
        )
        print(
            f"Y Range: {basic['spatial_bounds']['y_min']:.0f} to "
            f"{basic['spatial_bounds']['y_max']:.0f}"
        )

        print("\nDATA STATISTICS:")
        print(f"Min: {summary['min']:.4f}")
        print(f"Max: {summary['max']:.4f}")
        print(f"Mean: {summary['mean']:.4f}")
        print(f"Std: {summary['std']:.4f}")
        print(f"Missing: {summary['missing_percentage']:.2f}%")
        print("=" * 50 + "\n")

    # Temporal Analysis Methods
    def get_monthly_averages(
        self,
        months: list[int] | None = None,
        years: list[int] | None = None,
        method: str = "mean",
    ) -> xr.Dataset:
        """Get monthly averages.

        Parameters
        ----------
        months : list of int, optional
            Specific months to include (1-12)
        years : list of int, optional
            Specific years to include
        method : str
            Aggregation method

        Returns
        -------
        xr.Dataset
            Monthly aggregated data

        """
        return self.temporal_aggregator.monthly_aggregation(
            method=method, specific_months=months, years=years
        )

    def get_annual_averages(
        self, years: list[int] | None = None, method: str = "mean"
    ) -> xr.Dataset:
        """Get annual averages.

        Parameters
        ----------
        years : list of int, optional
            Specific years to include
        method : str
            Aggregation method

        Returns
        -------
        xr.Dataset
            Annual aggregated data

        """
        return self.temporal_aggregator.annual_aggregation(
            method=method, specific_years=years
        )

    def get_seasonal_averages(
        self,
        seasons: list[str] | None = None,
        years: list[int] | None = None,
        method: str = "mean",
    ) -> xr.Dataset:
        """Get seasonal averages.

        Parameters
        ----------
        seasons : list of str, optional
            Specific seasons ('winter', 'spring', 'summer', 'autumn')
        years : list of int, optional
            Specific years to include
        method : str
            Aggregation method

        Returns
        -------
        xr.Dataset
            Seasonal aggregated data

        """
        return self.temporal_aggregator.seasonal_aggregation(
            method=method, seasons=seasons, years=years
        )

    def get_custom_period_averages(
        self,
        time_periods: list[tuple[str, str]],
        method: str = "mean",
        period_names: list[str] | None = None,
    ) -> xr.Dataset:
        """Get averages for custom time periods.

        Parameters
        ----------
        time_periods : list of tuples
            List of (start_date, end_date) tuples
        method : str
            Aggregation method
        period_names : list of str, optional
            Names for each period

        Returns
        -------
        xr.Dataset
            Custom period aggregated data

        """
        return self.temporal_aggregator.custom_time_aggregation(
            time_periods=time_periods, method=method, period_names=period_names
        )

    # Spatial Extraction Methods
    def extract_at_points(
        self,
        locations: str | Path | list[tuple[float, float]] | gpd.GeoDataFrame,
        method: str = "nearest",
    ) -> xr.Dataset:
        """Extract data at specific point locations.

        Parameters
        ----------
        locations : various
            Point locations (file path, list of coordinates, or GeoDataFrame)
        method : str
            Extraction method

        Returns
        -------
        xr.Dataset
            Extracted point data

        """
        return self.spatial_extractor.extract_points(locations, method=method)

    def extract_for_polygons(
        self,
        polygons: str | Path | gpd.GeoDataFrame,
        aggregation_method: str = "mean",
    ) -> xr.Dataset:
        """Extract data within polygon boundaries.

        Parameters
        ----------
        polygons : various
            Polygon boundaries (file path or GeoDataFrame)
        aggregation_method : str
            Spatial aggregation method

        Returns
        -------
        xr.Dataset
            Extracted polygon data

        """
        return self.spatial_extractor.extract_polygons(polygons, aggregation_method)

    def extract_for_nuts3(
        self, nuts3_file: str | Path, aggregation_method: str = "mean"
    ) -> xr.Dataset:
        """Extract data for NUTS3 regions.

        Parameters
        ----------
        nuts3_file : str or Path
            Path to NUTS3 boundary file
        aggregation_method : str
            Spatial aggregation method

        Returns
        -------
        xr.Dataset
            Extracted NUTS3 data

        """
        return self.spatial_extractor.extract_nuts3_regions(
            nuts3_file, aggregation_method
        )

    def extract_with_event_dates(
        self,
        locations: str | Path | gpd.GeoDataFrame,
        date_column: str = "date",
        days_before: int = 0,
        aggregation_method: str = "mean",
    ) -> xr.Dataset:
        """Extract data at locations for specific event dates.

        Parameters
        ----------
        locations : various
            Locations with associated event dates
        date_column : str
            Column containing event dates
        days_before : int
            Number of days before each event to include
        aggregation_method : str
            Temporal aggregation method

        Returns
        -------
        xr.Dataset
            Extracted event-based data

        """
        return self.spatial_extractor.extract_with_dates(
            locations, date_column, days_before, aggregation_method
        )

    # Visualization Methods
    def plot_map(self, time_index: int | str = 0, **kwargs) -> Figure:
        """Create a spatial map."""
        return self.visualizer.plot_spatial_map(time_index=time_index, **kwargs)

    def plot_time_series(
        self, location: dict[str, float] | None = None, **kwargs
    ) -> Figure:
        """Create a time series plot."""
        return self.visualizer.plot_time_series(location=location, **kwargs)

    def plot_seasonal_cycle(self, **kwargs) -> Figure:
        """Create a seasonal cycle plot."""
        return self.visualizer.plot_seasonal_cycle(**kwargs)

    def plot_distribution(self, **kwargs) -> Figure:
        """Create a distribution plot."""
        return self.visualizer.plot_distribution(**kwargs)

    def plot_spatial_statistics(self, statistic: str = "mean", **kwargs) -> Figure:
        """Create a spatial statistics map."""
        return self.visualizer.plot_spatial_statistics(statistic=statistic, **kwargs)

    def create_animation(self, output_path: str | Path, **kwargs) -> None:
        """Create an animation of temporal evolution."""
        self.visualizer.create_animation(output_path, **kwargs)

    # Export Methods
    def export_to_netcdf(self, output_path: str | Path, **kwargs) -> None:
        """Export data to NetCDF format."""
        self.exporter.to_netcdf(output_path, **kwargs)

    def export_to_geotiff(self, output_path: str | Path, **kwargs) -> None:
        """Export data to GeoTIFF format."""
        self.exporter.to_geotiff(output_path, **kwargs)

    def export_to_csv(self, output_path: str | Path, **kwargs) -> None:
        """Export data to CSV format."""
        self.exporter.to_csv(output_path, **kwargs)

    # Comprehensive Analysis Workflows
    def analyze_temporal_patterns(
        self,
        output_dir: str | Path,
        save_plots: bool = True,
        save_data: bool = True,
    ) -> dict[str, Any]:
        """Perform comprehensive temporal pattern analysis.

        Parameters
        ----------
        output_dir : str or Path
            Output directory for results
        save_plots : bool
            Whether to save plots
        save_data : bool
            Whether to save processed data

        Returns
        -------
        dict
            Analysis results

        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {}

        # Monthly averages
        monthly_data = self.get_monthly_averages()
        results["monthly_averages"] = monthly_data

        # Annual averages
        annual_data = self.get_annual_averages()
        results["annual_averages"] = annual_data

        # Seasonal averages
        seasonal_data = self.get_seasonal_averages()
        results["seasonal_averages"] = seasonal_data

        if save_data:
            # Export aggregated data
            monthly_data.to_netcdf(output_dir / "monthly_averages.nc")
            annual_data.to_netcdf(output_dir / "annual_averages.nc")
            seasonal_data.to_netcdf(output_dir / "seasonal_averages.nc")

        if save_plots:
            # Create visualizations
            self.plot_time_series(save_path=output_dir / "time_series.png")
            self.plot_seasonal_cycle(save_path=output_dir / "seasonal_cycle.png")
            self.plot_distribution(save_path=output_dir / "distribution.png")

            # Spatial statistics maps
            for stat in ["mean", "max", "min", "std"]:
                self.plot_spatial_statistics(
                    statistic=stat, save_path=output_dir / f"spatial_{stat}.png"
                )

        logger.info(
            f"Temporal pattern analysis completed. Results saved to {output_dir}"
        )
        return results

    def analyze_spatial_patterns(
        self,
        regions_file: str | Path,
        output_dir: str | Path,
        save_plots: bool = True,
        save_data: bool = True,
        export_formats: list[str] | None = None,
    ) -> dict[str, Any]:
        """Perform comprehensive spatial pattern analysis.

        Parameters
        ----------
        regions_file : str or Path
            Path to spatial regions file
        output_dir : str or Path
            Output directory for results
        save_plots : bool
            Whether to save plots
        save_data : bool
            Whether to save processed data
        export_formats : list of str
            Export formats for extracted data

        Returns
        -------
        dict
            Analysis results

        """
        if export_formats is None:
            export_formats = ["csv", "geojson"]
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {}

        # Load regions
        regions_gdf = self.spatial_extractor.load_spatial_boundaries(regions_file)

        # Extract data for regions
        extracted_data = self.extract_for_polygons(regions_gdf)
        results["extracted_data"] = extracted_data

        if save_data:
            # Export extracted data to multiple formats
            export_paths = self.exporter.extracted_polygons_to_formats(
                extracted_data, regions_gdf, output_dir, export_formats
            )
            results["export_paths"] = export_paths

        if save_plots:
            # Create sample spatial maps
            self.plot_map(time_index=0, save_path=output_dir / "spatial_map_sample.png")
            self.plot_spatial_statistics(save_path=output_dir / "spatial_mean.png")

        logger.info(
            f"Spatial pattern analysis completed. Results saved to {output_dir}"
        )
        return results

    def comprehensive_analysis(
        self,
        output_dir: str | Path,
        regions_file: str | Path | None = None,
        points_file: str | Path | None = None,
    ) -> dict[str, Any]:
        """Perform comprehensive analysis including temporal and spatial patterns.

        Parameters
        ----------
        output_dir : str or Path
            Output directory for all results
        regions_file : str or Path, optional
            Path to spatial regions file
        points_file : str or Path, optional
            Path to point locations file

        Returns
        -------
        dict
            Complete analysis results

        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {"dataset_info": self.get_info()}

        # Temporal analysis
        temporal_dir = output_dir / "temporal_analysis"
        results["temporal_analysis"] = self.analyze_temporal_patterns(temporal_dir)

        # Spatial analysis for regions
        if regions_file:
            spatial_dir = output_dir / "spatial_analysis_regions"
            results["spatial_analysis_regions"] = self.analyze_spatial_patterns(
                regions_file, spatial_dir
            )

        # Point extraction
        if points_file:
            points_dir = output_dir / "point_extraction"
            points_dir.mkdir(parents=True, exist_ok=True)

            extracted_points = self.extract_at_points(points_file)
            results["point_extraction"] = extracted_points

            # Export point data
            self.exporter.extracted_points_to_formats(
                extracted_points, points_dir, ["csv", "geojson"]
            )

        # Create metadata file
        self.exporter.create_metadata_file(
            output_dir / "analysis_metadata.json",
            processing_info={
                "analysis_type": "comprehensive",
                "regions_file": str(regions_file) if regions_file else None,
                "points_file": str(points_file) if points_file else None,
            },
        )

        logger.info(f"Comprehensive analysis completed. Results saved to {output_dir}")
        return results

    def close(self) -> None:
        """Close the dataset and clean up resources."""
        if self.reader:
            self.reader.close()
        logger.info("PollutionAnalyzer closed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
