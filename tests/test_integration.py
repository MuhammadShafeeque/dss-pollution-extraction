"""
Integration tests for pollution_extraction library.

These tests verify that all components work together correctly in realistic workflows.
"""

import pytest
import numpy as np
import pandas as pd
import xarray as xr
import tempfile
import json
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

from pollution_extraction import PollutionAnalyzer
from pollution_extraction.core.data_reader import FileDataReader, CopernicusDataReader
from pollution_extraction.core.temporal_aggregator import (
    DailyAggregator,
    MonthlyAggregator,
)
from pollution_extraction.core.spatial_extractor import PointExtractor, RegionExtractor
from pollution_extraction.core.data_visualizer import SpatialPlotter, TemporalPlotter
from pollution_extraction.core.data_exporter import (
    CSVExporter,
    NetCDFExporter,
    GeoJSONExporter,
)

from tests import (
    create_sample_netcdf,
    create_sample_csv,
    create_sample_geojson,
    WHO_THRESHOLDS,
    EU_THRESHOLDS,
    setup_test_environment,
    cleanup_test_environment,
    TEMP_DIR,
)


class TestPollutionAnalyzerIntegration:
    """Test complete workflows using PollutionAnalyzer."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment and comprehensive test data."""
        setup_test_environment()

        # Create comprehensive European air quality dataset
        times = pd.date_range("2024-01-01", periods=365, freq="D")  # Full year
        lats = np.linspace(35, 70, 36)  # Europe coverage
        lons = np.linspace(-10, 40, 51)

        # Create realistic spatial and temporal patterns
        lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")

        # Urban pollution centers (lat, lon, intensity)
        urban_centers = [
            (52.5, 13.4, 35),  # Berlin
            (48.8, 2.3, 40),  # Paris
            (51.5, -0.1, 45),  # London
            (41.9, 12.5, 38),  # Rome
            (40.4, -3.7, 42),  # Madrid
            (55.7, 37.6, 30),  # Moscow
            (50.1, 8.7, 36),  # Frankfurt
            (45.5, 9.2, 34),  # Milan
            (59.9, 10.8, 28),  # Oslo
            (47.4, 19.1, 32),  # Budapest
        ]

        # Create base pollution field
        pm25_base = np.full_like(lat_grid, 12.0)  # Rural background
        no2_base = np.full_like(lat_grid, 18.0)
        o3_base = np.full_like(lat_grid, 80.0)

        # Add urban hotspots
        for city_lat, city_lon, intensity in urban_centers:
            distance = np.sqrt((lat_grid - city_lat) ** 2 + (lon_grid - city_lon) ** 2)
            urban_pm25 = intensity * np.exp(-(distance**2) / 4)
            urban_no2 = intensity * 1.3 * np.exp(-(distance**2) / 4)
            urban_o3 = (
                -intensity * 0.5 * np.exp(-(distance**2) / 4)
            )  # O3 titration in cities

            pm25_base += urban_pm25
            no2_base += urban_no2
            o3_base += urban_o3

        # Add topographic effects
        elevation_effect = 8 * np.exp(-((lat_grid - 47) ** 2) / 50)  # Alpine effect
        pm25_base += elevation_effect

        # Create temporal patterns
        pm25_data = np.zeros((len(times), len(lats), len(lons)))
        no2_data = np.zeros((len(times), len(lats), len(lons)))
        o3_data = np.zeros((len(times), len(lats), len(lons)))

        for i, time in enumerate(times):
            # Seasonal variation
            day_of_year = time.dayofyear
            seasonal_factor = 1 + 0.4 * np.sin((day_of_year - 80) * 2 * np.pi / 365)
            winter_heating = 1 + 0.3 * np.exp(
                -((day_of_year - 15) ** 2 + (day_of_year - 350) ** 2) / 1000
            )

            # Weekly variation (lower on weekends)
            weekly_factor = 1 - 0.2 * (time.weekday >= 5)

            # Add meteorological influence (simplified)
            weather_factor = 1 + 0.2 * np.sin(i * 2 * np.pi / 10)  # 10-day cycle

            pm25_data[i] = (
                pm25_base
                * seasonal_factor
                * winter_heating
                * weekly_factor
                * weather_factor
            )
            no2_data[i] = no2_base * seasonal_factor * weekly_factor * weather_factor
            o3_data[i] = (
                o3_base * (2 - seasonal_factor) * weather_factor
            )  # Higher in summer

            # Add noise
            pm25_data[i] += np.random.normal(0, 2, pm25_data[i].shape)
            no2_data[i] += np.random.normal(0, 3, no2_data[i].shape)
            o3_data[i] += np.random.normal(0, 5, o3_data[i].shape)

        # Ensure non-negative values
        pm25_data = np.maximum(pm25_data, 0)
        no2_data = np.maximum(no2_data, 0)
        o3_data = np.maximum(o3_data, 0)

        # Create comprehensive xarray Dataset
        self.comprehensive_dataset = xr.Dataset(
            {
                "pm25": (
                    ["time", "latitude", "longitude"],
                    pm25_data,
                    {
                        "units": "μg/m³",
                        "long_name": "PM2.5 mass concentration",
                        "standard_name": "mass_concentration_of_pm2p5_ambient_aerosol_particles_in_air",
                    },
                ),
                "no2": (
                    ["time", "latitude", "longitude"],
                    no2_data,
                    {
                        "units": "μg/m³",
                        "long_name": "NO2 mass concentration",
                        "standard_name": "mass_concentration_of_nitrogen_dioxide_in_air",
                    },
                ),
                "o3": (
                    ["time", "latitude", "longitude"],
                    o3_data,
                    {
                        "units": "μg/m³",
                        "long_name": "O3 mass concentration",
                        "standard_name": "mass_concentration_of_ozone_in_air",
                    },
                ),
            },
            coords={
                "time": ("time", times, {"long_name": "time", "standard_name": "time"}),
                "latitude": (
                    "latitude",
                    lats,
                    {
                        "units": "degrees_north",
                        "long_name": "latitude",
                        "standard_name": "latitude",
                    },
                ),
                "longitude": (
                    "longitude",
                    lons,
                    {
                        "units": "degrees_east",
                        "long_name": "longitude",
                        "standard_name": "longitude",
                    },
                ),
            },
            attrs={
                "title": "European Air Quality Dataset - Test Data",
                "institution": "pollution_extraction test suite",
                "source": "Synthetic data for testing",
                "conventions": "CF-1.8",
                "created": "2024-01-01",
                "description": "Comprehensive synthetic air quality dataset covering Europe for one year",
            },
        )

        # Save dataset
        self.data_file = TEMP_DIR / "comprehensive_europe.nc"
        self.comprehensive_dataset.to_netcdf(self.data_file)

        # Create test points (major European cities)
        self.test_cities = pd.DataFrame(
            {
                "city": [
                    "Berlin",
                    "Paris",
                    "London",
                    "Rome",
                    "Madrid",
                    "Moscow",
                    "Amsterdam",
                    "Vienna",
                ],
                "country": [
                    "Germany",
                    "France",
                    "UK",
                    "Italy",
                    "Spain",
                    "Russia",
                    "Netherlands",
                    "Austria",
                ],
                "latitude": [52.5, 48.8, 51.5, 41.9, 40.4, 55.7, 52.4, 48.2],
                "longitude": [13.4, 2.3, -0.1, 12.5, -3.7, 37.6, 4.9, 16.4],
                "population": [
                    3700000,
                    2200000,
                    9000000,
                    2870000,
                    3200000,
                    12500000,
                    850000,
                    1900000,
                ],
            }
        )

        # Create test regions
        self.test_regions = [
            {"name": "Central Europe", "geometry": self._create_bbox(8, 45, 20, 55)},
            {"name": "Western Europe", "geometry": self._create_bbox(-5, 45, 8, 55)},
            {"name": "Nordic Countries", "geometry": self._create_bbox(5, 55, 30, 70)},
            {"name": "Mediterranean", "geometry": self._create_bbox(-5, 35, 20, 45)},
        ]

        yield
        cleanup_test_environment()

    def _create_bbox(self, lon_min, lat_min, lon_max, lat_max):
        """Create bounding box geometry."""
        from shapely.geometry import box

        return box(lon_min, lat_min, lon_max, lat_max)

    def test_full_analysis_workflow(self):
        """Test complete analysis workflow from data loading to export."""
        # Initialize analyzer
        analyzer = PollutionAnalyzer()

        # Step 1: Load data
        analyzer.load_data(str(self.data_file))
        assert analyzer.data is not None
        assert "pm25" in analyzer.data.variables

        # Step 2: Extract data for major cities
        city_data = analyzer.extract_points(
            points=self.test_cities[["latitude", "longitude"]].values.tolist(),
            variables=["pm25", "no2", "o3"],
        )

        assert isinstance(city_data, pd.DataFrame)
        assert len(city_data) > 0
        assert "pm25" in city_data.columns

        # Step 3: Extract regional data
        regional_data = analyzer.extract_regions(
            regions=self.test_regions,
            variables=["pm25", "no2"],
            aggregation_method="mean",
        )

        assert isinstance(regional_data, pd.DataFrame)
        assert len(regional_data) > 0
        assert "region_name" in regional_data.columns

        # Step 4: Temporal aggregation
        monthly_data = analyzer.aggregate_temporal(
            frequency="monthly", method="mean", variables=["pm25", "no2"]
        )

        assert isinstance(monthly_data, xr.Dataset)
        assert len(monthly_data.time) == 12  # 12 months

        # Step 5: Health threshold analysis
        threshold_analysis = analyzer.analyze_health_thresholds(
            pollutant="pm25", threshold_type="WHO"
        )

        assert isinstance(threshold_analysis, xr.Dataset)
        assert "exceedance_days" in threshold_analysis.variables

        # Step 6: Export results
        city_output = TEMP_DIR / "city_analysis.csv"
        regional_output = TEMP_DIR / "regional_analysis.csv"
        monthly_output = TEMP_DIR / "monthly_data.nc"

        analyzer.export_data(city_data, city_output, format="csv")
        analyzer.export_data(regional_data, regional_output, format="csv")
        analyzer.export_data(monthly_data, monthly_output, format="netcdf")

        assert city_output.exists()
        assert regional_output.exists()
        assert monthly_output.exists()

    def test_air_quality_assessment_workflow(self):
        """Test air quality assessment workflow with health analysis."""
        analyzer = PollutionAnalyzer()
        analyzer.load_data(str(self.data_file))

        # Extract data for major cities
        city_coords = self.test_cities[["latitude", "longitude"]].values.tolist()
        city_names = self.test_cities["city"].tolist()

        # Create named points
        named_points = [
            {"lat": coord[0], "lon": coord[1], "name": name}
            for coord, name in zip(city_coords, city_names)
        ]

        city_data = analyzer.extract_points(
            points=named_points, variables=["pm25", "no2", "o3"]
        )

        # Annual mean analysis
        annual_means = analyzer.aggregate_temporal(
            frequency="annual", method="mean", variables=["pm25", "no2", "o3"]
        )

        # WHO threshold analysis for each pollutant
        pollutants = ["pm25", "no2"]
        threshold_results = {}

        for pollutant in pollutants:
            threshold_results[pollutant] = analyzer.analyze_health_thresholds(
                pollutant=pollutant, threshold_type="WHO"
            )

        # Verify health threshold analysis
        for pollutant, result in threshold_results.items():
            assert "exceedance_days" in result.variables
            assert "exceedance_percentage" in result.variables
            assert "annual_mean" in result.variables

            # Check that exceedance days are reasonable (0-365)
            exceedance_days = result["exceedance_days"]
            assert (exceedance_days >= 0).all()
            assert (exceedance_days <= 365).all()

        # Verify that cities show expected pollution patterns
        # (Urban areas should generally have higher pollution)
        city_means = city_data.groupby("name")["pm25"].mean()

        # Berlin, Paris, London should be among higher values
        major_cities = ["Berlin", "Paris", "London"]
        major_city_means = [
            city_means.get(city, 0) for city in major_cities if city in city_means.index
        ]

        if major_city_means:
            overall_mean = city_data["pm25"].mean()
            assert (
                np.mean(major_city_means) > overall_mean * 0.8
            )  # At least 80% of overall mean

    def test_comparative_analysis_workflow(self):
        """Test comparative analysis between regions and time periods."""
        analyzer = PollutionAnalyzer()
        analyzer.load_data(str(self.data_file))

        # Compare different regions
        regional_comparison = {}
        for region in self.test_regions:
            regional_data = analyzer.extract_regions(
                regions=[region], variables=["pm25", "no2"], aggregation_method="mean"
            )
            regional_comparison[region["name"]] = regional_data

        # Verify regional differences
        regional_means = {}
        for region_name, data in regional_comparison.items():
            regional_means[region_name] = data["pm25"].mean()

        # Should have reasonable variation between regions
        mean_values = list(regional_means.values())
        if len(mean_values) > 1:
            coefficient_of_variation = np.std(mean_values) / np.mean(mean_values)
            assert coefficient_of_variation > 0.05  # At least 5% variation

        # Seasonal comparison
        seasonal_data = {}
        seasons = {
            "Winter": [12, 1, 2],
            "Spring": [3, 4, 5],
            "Summer": [6, 7, 8],
            "Autumn": [9, 10, 11],
        }

        for season_name, months in seasons.items():
            seasonal_subset = analyzer.data.sel(
                time=analyzer.data.time.dt.month.isin(months)
            )
            seasonal_mean = seasonal_subset.mean(dim="time")
            seasonal_data[season_name] = seasonal_mean

        # Verify seasonal patterns
        # Winter should generally have higher PM2.5 (heating)
        # Summer should have higher O3 (photochemical formation)
        winter_pm25 = seasonal_data["Winter"]["pm25"].mean().values
        summer_pm25 = seasonal_data["Summer"]["pm25"].mean().values
        winter_o3 = seasonal_data["Winter"]["o3"].mean().values
        summer_o3 = seasonal_data["Summer"]["o3"].mean().values

        # These are tendencies, not strict rules, so use loose thresholds
        assert winter_pm25 > summer_pm25 * 0.8  # Winter PM2.5 at least 80% of summer
        assert summer_o3 > winter_o3 * 0.9  # Summer O3 at least 90% of winter

    def test_performance_large_dataset(self):
        """Test performance with large dataset operations."""
        analyzer = PollutionAnalyzer()

        start_time = time.time()

        # Load large dataset
        analyzer.load_data(str(self.data_file))
        load_time = time.time() - start_time

        # Extract many points
        start_extract = time.time()

        # Create 50 random points across Europe
        n_points = 50
        random_points = [
            (np.random.uniform(35, 70), np.random.uniform(-10, 40))
            for _ in range(n_points)
        ]

        point_data = analyzer.extract_points(
            points=random_points, variables=["pm25", "no2"]
        )

        extract_time = time.time() - start_extract

        # Temporal aggregation
        start_aggregate = time.time()
        daily_data = analyzer.aggregate_temporal(
            frequency="daily", method="mean", variables=["pm25"]
        )
        aggregate_time = time.time() - start_aggregate

        # Performance assertions (generous limits for CI environments)
        assert load_time < 30  # Load should complete in 30 seconds
        assert extract_time < 60  # Extract 50 points in 60 seconds
        assert aggregate_time < 120  # Daily aggregation in 2 minutes

        # Verify results
        assert len(point_data) == n_points * len(analyzer.data.time)
        assert len(daily_data.time) == 365

    def test_data_quality_validation_workflow(self):
        """Test data quality validation and handling."""
        analyzer = PollutionAnalyzer()
        analyzer.load_data(str(self.data_file))

        # Check data completeness
        completeness = analyzer.check_data_completeness()

        assert isinstance(completeness, dict)
        assert "pm25" in completeness
        assert "no2" in completeness

        # All synthetic data should be complete
        for variable, completeness_pct in completeness.items():
            assert completeness_pct >= 99.0  # At least 99% complete

        # Check data ranges
        data_stats = analyzer.get_data_statistics()

        assert isinstance(data_stats, dict)
        assert "pm25" in data_stats

        # PM2.5 values should be realistic
        pm25_stats = data_stats["pm25"]
        assert pm25_stats["min"] >= 0  # Non-negative
        assert pm25_stats["max"] < 200  # Reasonable maximum
        assert pm25_stats["mean"] > 5  # Reasonable minimum mean

        # Check temporal consistency
        temporal_stats = analyzer.analyze_temporal_patterns()

        assert isinstance(temporal_stats, dict)
        assert "seasonal_variation" in temporal_stats
        assert "trend_analysis" in temporal_stats

    def test_multi_format_input_output_workflow(self):
        """Test workflow with multiple input and output formats."""
        analyzer = PollutionAnalyzer()

        # Test NetCDF input
        analyzer.load_data(str(self.data_file))

        # Extract point data
        point_data = analyzer.extract_points(
            points=[(52.5, 13.4), (48.8, 2.3)], variables=["pm25", "no2"]
        )

        # Export to multiple formats
        csv_output = TEMP_DIR / "multiformat_test.csv"
        excel_output = TEMP_DIR / "multiformat_test.xlsx"
        geojson_output = TEMP_DIR / "multiformat_test.geojson"

        # CSV export
        analyzer.export_data(point_data, csv_output, format="csv")
        assert csv_output.exists()

        # Excel export
        analyzer.export_data(point_data, excel_output, format="excel")
        assert excel_output.exists()

        # GeoJSON export (add coordinates to point_data first)
        point_geo_data = point_data.copy()
        if "latitude" not in point_geo_data.columns:
            # Add coordinates for GeoJSON export
            coords = [(52.5, 13.4), (48.8, 2.3)]
            n_times = len(point_data) // len(coords)

            lats = np.tile([coord[0] for coord in coords], n_times)
            lons = np.tile([coord[1] for coord in coords], n_times)

            point_geo_data["latitude"] = lats
            point_geo_data["longitude"] = lons

        analyzer.export_data(
            point_geo_data,
            geojson_output,
            format="geojson",
            lat_column="latitude",
            lon_column="longitude",
        )
        assert geojson_output.exists()

        # Verify file contents
        reimported_csv = pd.read_csv(csv_output)
        assert len(reimported_csv) == len(point_data)
        assert "pm25" in reimported_csv.columns


class TestComponentIntegration:
    """Test integration between different library components."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        setup_test_environment()

        # Create test data
        self.test_data = create_sample_netcdf()
        self.data_file = TEMP_DIR / "component_test.nc"
        self.test_data.to_netcdf(self.data_file)

        yield
        cleanup_test_environment()

    def test_reader_aggregator_integration(self):
        """Test integration between data reader and temporal aggregator."""
        # Read data
        reader = FileDataReader(str(self.data_file))
        data = reader.read_data()

        # Aggregate temporally
        aggregator = DailyAggregator(method="mean")
        daily_data = aggregator.aggregate(data)

        # Should produce valid aggregated data
        assert isinstance(daily_data, xr.Dataset)
        assert len(daily_data.time) <= len(data.time)
        assert "pm25" in daily_data.variables

    def test_extractor_visualizer_integration(self):
        """Test integration between spatial extractor and visualizer."""
        # Extract points
        extractor = PointExtractor()
        points = [(52.5, 13.4), (48.8, 2.3)]

        reader = FileDataReader(str(self.data_file))
        data = reader.read_data()

        point_data = extractor.extract(data, points)

        # Visualize extracted data
        plotter = TemporalPlotter()
        fig = plotter.plot_timeseries(
            point_data, x_col="time", y_col="pm25", title="Extracted Point Time Series"
        )

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_aggregator_exporter_integration(self):
        """Test integration between aggregator and exporter."""
        # Load and aggregate data
        reader = FileDataReader(str(self.data_file))
        data = reader.read_data()

        aggregator = MonthlyAggregator(method="mean")
        monthly_data = aggregator.aggregate(data)

        # Export aggregated data
        exporter = NetCDFExporter()
        output_file = TEMP_DIR / "aggregated_export.nc"

        result = exporter.export(monthly_data, output_file)

        assert result is True
        assert output_file.exists()

        # Verify exported data
        reimported = xr.open_dataset(output_file)
        assert "pm25" in reimported.variables
        assert len(reimported.time) == len(monthly_data.time)
        reimported.close()

    def test_complete_processing_chain(self):
        """Test complete processing chain with all components."""
        # 1. Read data
        reader = FileDataReader(str(self.data_file))
        original_data = reader.read_data()

        # 2. Extract regions
        from shapely.geometry import box

        regions = [{"name": "Test Region", "geometry": box(10, 40, 20, 50)}]

        region_extractor = RegionExtractor()
        regional_data = region_extractor.extract(original_data, regions)

        # 3. Temporal aggregation
        aggregator = DailyAggregator(method="mean")
        aggregated_data = aggregator.aggregate(original_data)

        # 4. Visualization
        spatial_plotter = SpatialPlotter()
        spatial_fig = spatial_plotter.plot_contour(
            aggregated_data["pm25"].isel(time=0), title="Daily Mean PM2.5"
        )

        # 5. Export results
        csv_exporter = CSVExporter()
        regional_output = TEMP_DIR / "chain_regional.csv"
        csv_exporter.export(regional_data, regional_output)

        netcdf_exporter = NetCDFExporter()
        aggregated_output = TEMP_DIR / "chain_aggregated.nc"
        netcdf_exporter.export(aggregated_data, aggregated_output)

        # Verify all steps completed successfully
        assert isinstance(regional_data, pd.DataFrame)
        assert isinstance(aggregated_data, xr.Dataset)
        assert spatial_fig is not None
        assert regional_output.exists()
        assert aggregated_output.exists()

        import matplotlib.pyplot as plt

        plt.close(spatial_fig)


class TestErrorHandlingIntegration:
    """Test error handling across integrated workflows."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        setup_test_environment()
        yield
        cleanup_test_environment()

    def test_invalid_data_handling(self):
        """Test handling of invalid data throughout workflow."""
        analyzer = PollutionAnalyzer()

        # Test loading invalid file
        invalid_file = TEMP_DIR / "invalid.txt"
        with open(invalid_file, "w") as f:
            f.write("This is not valid data")

        with pytest.raises(Exception):
            analyzer.load_data(str(invalid_file))

    def test_missing_variable_handling(self):
        """Test handling of missing variables in workflow."""
        # Create data without certain variables
        times = pd.date_range("2024-01-01", periods=5, freq="D")
        lats = np.linspace(50, 52, 3)
        lons = np.linspace(10, 12, 3)

        # Only PM2.5, no NO2
        pm25_data = np.random.uniform(10, 30, (len(times), len(lats), len(lons)))

        limited_data = xr.Dataset(
            {"pm25": (["time", "latitude", "longitude"], pm25_data)},
            coords={"time": times, "latitude": lats, "longitude": lons},
        )

        data_file = TEMP_DIR / "limited_data.nc"
        limited_data.to_netcdf(data_file)

        analyzer = PollutionAnalyzer()
        analyzer.load_data(str(data_file))

        # Should handle request for missing variable gracefully
        with pytest.raises(KeyError):
            analyzer.extract_points(
                points=[(51, 11)],
                variables=["pm25", "no2"],  # no2 doesn't exist
            )

    def test_empty_result_handling(self):
        """Test handling of empty results in workflow."""
        analyzer = PollutionAnalyzer()

        # Create minimal data
        data = create_sample_netcdf()
        data_file = TEMP_DIR / "minimal_data.nc"
        data.to_netcdf(data_file)

        analyzer.load_data(str(data_file))

        # Extract points far outside domain
        outside_points = [(90, 180), (-90, -180)]  # North pole, opposite

        # Should handle gracefully (return empty or NaN results)
        try:
            result = analyzer.extract_points(points=outside_points, variables=["pm25"])
            # If successful, should return empty or NaN data
            if not result.empty:
                assert result["pm25"].isna().all()
        except Exception as e:
            # Should be a meaningful error message
            assert len(str(e)) > 0


class TestScalabilityIntegration:
    """Test scalability of integrated workflows."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        setup_test_environment()
        yield
        cleanup_test_environment()

    def test_large_point_extraction(self):
        """Test extraction of many points."""
        # Create test data
        data = create_sample_netcdf()
        data_file = TEMP_DIR / "scalability_test.nc"
        data.to_netcdf(data_file)

        analyzer = PollutionAnalyzer()
        analyzer.load_data(str(data_file))

        # Generate many random points
        n_points = 1000
        random_points = [
            (np.random.uniform(40, 60), np.random.uniform(-10, 30))
            for _ in range(n_points)
        ]

        start_time = time.time()

        point_data = analyzer.extract_points(points=random_points, variables=["pm25"])

        end_time = time.time()

        # Should complete in reasonable time
        assert (end_time - start_time) < 120  # 2 minutes for 1000 points
        assert len(point_data) > 0

    def test_batch_processing_simulation(self):
        """Test simulation of batch processing workflow."""
        # Create multiple small datasets
        datasets = []
        for i in range(5):
            data = create_sample_netcdf()
            file_path = TEMP_DIR / f"batch_data_{i}.nc"
            data.to_netcdf(file_path)
            datasets.append(file_path)

        # Process each dataset
        all_results = []

        for dataset_file in datasets:
            analyzer = PollutionAnalyzer()
            analyzer.load_data(str(dataset_file))

            # Extract standard points
            points = [(52.5, 13.4), (48.8, 2.3)]
            result = analyzer.extract_points(points=points, variables=["pm25"])

            all_results.append(result)

        # Combine results
        combined_results = pd.concat(all_results, ignore_index=True)

        assert len(combined_results) > 0
        assert len(combined_results) == sum(len(r) for r in all_results)

    def test_memory_usage_monitoring(self):
        """Test memory usage in complex workflows."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Create and process moderately large dataset
        times = pd.date_range("2024-01-01", periods=100, freq="D")
        lats = np.linspace(40, 60, 51)  # 51 points
        lons = np.linspace(-10, 30, 81)  # 81 points

        # Create data (100 * 51 * 81 = ~400k points)
        pm25_data = np.random.uniform(10, 40, (len(times), len(lats), len(lons)))

        large_data = xr.Dataset(
            {"pm25": (["time", "latitude", "longitude"], pm25_data)},
            coords={"time": times, "latitude": lats, "longitude": lons},
        )

        data_file = TEMP_DIR / "memory_test.nc"
        large_data.to_netcdf(data_file)

        # Process the large dataset
        analyzer = PollutionAnalyzer()
        analyzer.load_data(str(data_file))

        # Multiple operations
        points = [(52.5, 13.4), (48.8, 2.3), (41.9, 12.5)]
        point_data = analyzer.extract_points(points=points, variables=["pm25"])

        daily_data = analyzer.aggregate_temporal(
            frequency="daily", method="mean", variables=["pm25"]
        )

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (< 500MB for this test)
        assert memory_increase < 500

        # Clean up
        del large_data
        del daily_data
        del point_data


class TestRealWorldScenarios:
    """Test realistic usage scenarios."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        setup_test_environment()
        yield
        cleanup_test_environment()

    def test_air_quality_monitoring_scenario(self):
        """Test air quality monitoring station analysis scenario."""
        # Create monitoring station data
        stations = pd.DataFrame(
            {
                "station_id": ["ST001", "ST002", "ST003", "ST004"],
                "station_name": [
                    "Berlin Center",
                    "Berlin Suburb",
                    "Paris Center",
                    "Paris Suburb",
                ],
                "latitude": [52.5, 52.6, 48.8, 48.9],
                "longitude": [13.4, 13.5, 2.3, 2.4],
                "station_type": ["urban", "suburban", "urban", "suburban"],
                "elevation": [34, 45, 35, 78],
            }
        )

        # Create gridded data
        data = create_sample_netcdf()
        data_file = TEMP_DIR / "monitoring_scenario.nc"
        data.to_netcdf(data_file)

        analyzer = PollutionAnalyzer()
        analyzer.load_data(str(data_file))

        # Extract data at monitoring stations
        station_coords = stations[["latitude", "longitude"]].values.tolist()
        station_data = analyzer.extract_points(
            points=station_coords, variables=["pm25", "no2"]
        )

        # Add station metadata
        n_times = len(station_data) // len(stations)
        station_info = pd.concat([stations] * n_times, ignore_index=True)

        # Merge if indices align
        if len(station_data) == len(station_info):
            combined_data = pd.concat([station_data, station_info], axis=1)
        else:
            combined_data = station_data

        # Analyze by station type
        if "station_type" in combined_data.columns:
            urban_data = combined_data[combined_data["station_type"] == "urban"]
            suburban_data = combined_data[combined_data["station_type"] == "suburban"]

            # Urban stations should generally have higher pollution
            if len(urban_data) > 0 and len(suburban_data) > 0:
                urban_mean = urban_data["pm25"].mean()
                suburban_mean = suburban_data["pm25"].mean()

                # This is a tendency, not a strict rule
                assert urban_mean >= suburban_mean * 0.8

        # Export station analysis
        output_file = TEMP_DIR / "station_analysis.csv"
        analyzer.export_data(combined_data, output_file, format="csv")
        assert output_file.exists()

    def test_policy_assessment_scenario(self):
        """Test policy impact assessment scenario."""
        # Create data representing before/after policy implementation
        data = create_sample_netcdf()
        data_file = TEMP_DIR / "policy_scenario.nc"
        data.to_netcdf(data_file)

        analyzer = PollutionAnalyzer()
        analyzer.load_data(str(data_file))

        # Define policy implementation date (middle of time series)
        policy_date = pd.Timestamp("2024-01-05")

        # Split data into before and after periods
        before_data = analyzer.data.sel(time=analyzer.data.time < policy_date)
        after_data = analyzer.data.sel(time=analyzer.data.time >= policy_date)

        # Calculate means for each period
        before_mean = before_data.mean(dim="time")
        after_mean = after_data.mean(dim="time")

        # Calculate change
        pollution_change = after_mean - before_mean

        # Export change analysis
        change_file = TEMP_DIR / "policy_change.nc"
        exporter = NetCDFExporter()
        exporter.export(pollution_change, change_file)

        assert change_file.exists()

        # Verify change data
        change_data = xr.open_dataset(change_file)
        assert "pm25" in change_data.variables
        assert "no2" in change_data.variables
        change_data.close()

    def test_health_impact_assessment_scenario(self):
        """Test health impact assessment scenario."""
        # Create population-weighted exposure analysis
        data = create_sample_netcdf()
        data_file = TEMP_DIR / "health_scenario.nc"
        data.to_netcdf(data_file)

        analyzer = PollutionAnalyzer()
        analyzer.load_data(str(data_file))

        # Define population centers with population data
        population_centers = pd.DataFrame(
            {
                "city": ["Berlin", "Paris", "Rome"],
                "latitude": [52.5, 48.8, 41.9],
                "longitude": [13.4, 2.3, 12.5],
                "population": [3700000, 2200000, 2870000],
            }
        )

        # Extract pollution data for population centers
        center_coords = population_centers[["latitude", "longitude"]].values.tolist()
        pollution_data = analyzer.extract_points(
            points=center_coords, variables=["pm25", "no2"]
        )

        # Calculate population-weighted exposure
        if len(pollution_data) >= len(population_centers):
            # Simple approach: assume first n rows correspond to cities
            n_times = len(pollution_data) // len(population_centers)

            weighted_exposure = {}
            for pollutant in ["pm25", "no2"]:
                total_exposure = 0
                total_population = 0

                for i, (_, city_info) in enumerate(population_centers.iterrows()):
                    city_pollution = pollution_data.iloc[
                        i * n_times : (i + 1) * n_times
                    ][pollutant].mean()
                    city_population = city_info["population"]

                    total_exposure += city_pollution * city_population
                    total_population += city_population

                weighted_exposure[pollutant] = total_exposure / total_population

            # Verify weighted exposure calculation
            assert "pm25" in weighted_exposure
            assert "no2" in weighted_exposure
            assert weighted_exposure["pm25"] > 0
            assert weighted_exposure["no2"] > 0

        # Health threshold analysis
        threshold_analysis = analyzer.analyze_health_thresholds(
            pollutant="pm25", threshold_type="WHO"
        )

        assert "exceedance_days" in threshold_analysis.variables

        # Export health assessment
        health_output = TEMP_DIR / "health_assessment.nc"
        exporter = NetCDFExporter()
        exporter.export(threshold_analysis, health_output)

        assert health_output.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
