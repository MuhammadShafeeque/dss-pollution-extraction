"""Example usage scripts for DSS pollution data analysis."""

import geopandas as gpd
import pandas as pd

from pollution_extraction import PollutionAnalyzer


def example_basic_analysis():
    """Basic analysis example."""
    print("=" * 60)
    print("BASIC POLLUTION DATA ANALYSIS EXAMPLE")
    print("=" * 60)

    # Initialize analyzer (replace with your actual file path)
    file_path = "path/to/your/pollution_data.nc"

    try:
        with PollutionAnalyzer(file_path, pollution_type="pm25") as analyzer:
            # Print dataset summary
            analyzer.print_summary()

            # Get basic statistics
            info = analyzer.get_info()
            print(f"Data covers {info['basic_info']['total_time_steps']} time steps")
            print(f"Mean concentration: {info['data_summary']['mean']:.3f} μg/m³")

            # Create a simple time series plot
            analyzer.plot_time_series(
                title="PM2.5 Time Series (Domain Average)",
                save_path="pm25_timeseries.png",
            )
            print("Time series plot saved as 'pm25_timeseries.png'")

            # Create a spatial map for the first time step
            analyzer.plot_map(
                time_index=0,
                title="PM2.5 Spatial Distribution",
                save_path="pm25_spatial.png",
            )
            print("Spatial map saved as 'pm25_spatial.png'")

    except FileNotFoundError:
        print(f"File not found: {file_path}")
        print("Please update the file_path variable with your actual NetCDF file path")


def example_temporal_analysis():
    """Temporal analysis example."""
    print("=" * 60)
    print("TEMPORAL ANALYSIS EXAMPLE")
    print("=" * 60)

    file_path = "path/to/your/pollution_data.nc"

    try:
        with PollutionAnalyzer(file_path, pollution_type="no2") as analyzer:
            # Monthly averages
            monthly_avg = analyzer.get_monthly_averages()
            print(f"Monthly averages calculated for {len(monthly_avg.month)} months")

            # Seasonal averages
            analyzer.get_seasonal_averages()
            print("Seasonal averages calculated")

            # Annual averages for specific years
            analyzer.get_annual_averages(years=[2010, 2011, 2012])
            print("Annual averages calculated for selected years")

            # Custom period analysis
            custom_periods = [
                ("2010-06-01", "2010-08-31"),  # Summer 2010
                ("2010-12-01", "2011-02-28"),  # Winter 2010-2011
            ]
            analyzer.get_custom_period_averages(
                custom_periods, period_names=["Summer_2010", "Winter_2010_2011"]
            )
            print("Custom period averages calculated")

            # Create seasonal cycle plot
            analyzer.plot_seasonal_cycle(
                title="NO2 Seasonal Cycle", save_path="no2_seasonal_cycle.png"
            )
            print("Seasonal cycle plot saved")

            # Export temporal data
            monthly_avg.to_netcdf("no2_monthly_averages.nc")
            print("Monthly averages exported to NetCDF")

    except FileNotFoundError:
        print(f"File not found: {file_path}")


def example_spatial_extraction():
    """Spatial extraction example."""
    print("=" * 60)
    print("SPATIAL EXTRACTION EXAMPLE")
    print("=" * 60)

    file_path = "path/to/your/pollution_data.nc"

    try:
        with PollutionAnalyzer(file_path, pollution_type="pm25") as analyzer:
            # Extract data at specific points
            # Example with coordinate pairs (LAEA projection coordinates)
            point_locations = [
                (4321000, 3210000),  # Example coordinates
                (4500000, 3400000),
                (4200000, 3100000),
            ]

            point_data = analyzer.extract_at_points(point_locations)
            print(f"Data extracted for {len(point_locations)} points")

            # Example with polygon extraction (if you have a shapefile)
            # polygon_file = "path/to/your/regions.shp"
            # polygon_data = analyzer.extract_for_polygons(polygon_file)
            # print("Data extracted for polygon regions")

            # Example with NUTS3 regions (if you have NUTS3 shapefile)
            # nuts3_file = "path/to/nuts3_regions.shp"
            # nuts3_data = analyzer.extract_for_nuts3(nuts3_file)
            # print("Data extracted for NUTS3 regions")

            # Export point data to CSV
            analyzer.exporter.extracted_points_to_formats(
                point_data, output_dir="extracted_data", formats=["csv"]
            )
            print("Point extraction data exported to CSV")

    except FileNotFoundError:
        print(f"File not found: {file_path}")


def example_event_based_analysis():
    """Event-based analysis example."""
    print("=" * 60)
    print("EVENT-BASED ANALYSIS EXAMPLE")
    print("=" * 60)

    file_path = "path/to/your/pollution_data.nc"

    try:
        # Create example event data
        events_data = pd.DataFrame(
            {
                "location_id": ["Site_A", "Site_B", "Site_C"],
                "lon": [10.0, 12.0, 8.0],  # Longitude (will be converted to LAEA)
                "lat": [52.0, 54.0, 50.0],  # Latitude (will be converted to LAEA)
                "date": ["2010-07-15", "2010-08-20", "2010-09-10"],
                "event_type": ["wildfire", "festival", "construction"],
            }
        )

        # Convert to GeoDataFrame
        from shapely.geometry import Point

        geometry = [
            Point(lon, lat)
            for lon, lat in zip(events_data["lon"], events_data["lat"], strict=False)
        ]
        events_gdf = gpd.GeoDataFrame(events_data, geometry=geometry, crs="EPSG:4326")

        # Save to file for the example
        events_gdf.to_file("example_events.geojson", driver="GeoJSON")

        with PollutionAnalyzer(file_path, pollution_type="pm25") as analyzer:
            # Extract data for events with 7 days before each event
            event_data = analyzer.extract_with_event_dates(
                "example_events.geojson",
                date_column="date",
                days_before=7,
                aggregation_method="mean",
            )
            print("Event-based data extraction completed")

            # Export event data
            analyzer.exporter.time_series_to_formats(
                event_data, output_dir="event_analysis", formats=["csv"]
            )
            print("Event analysis data exported")

    except FileNotFoundError:
        print(f"File not found: {file_path}")


def example_comprehensive_analysis():
    """Comprehensive analysis workflow."""
    print("=" * 60)
    print("COMPREHENSIVE ANALYSIS EXAMPLE")
    print("=" * 60)

    file_path = "path/to/your/pollution_data.nc"
    output_dir = "comprehensive_analysis_output"

    try:
        with PollutionAnalyzer(file_path, pollution_type="no2") as analyzer:
            # Run comprehensive analysis
            results = analyzer.comprehensive_analysis(
                output_dir=output_dir,
                # regions_file="path/to/regions.shp",  # Uncomment if you have regions
                # points_file="path/to/points.csv"     # Uncomment if you have points
            )

            print("Comprehensive analysis completed")
            print(f"Results saved to: {output_dir}")
            print(f"Available result keys: {list(results.keys())}")

            # Access specific results
            if "temporal_analysis" in results:
                temporal_results = results["temporal_analysis"]
                print(f"Temporal analysis included {len(temporal_results)} components")

    except FileNotFoundError:
        print(f"File not found: {file_path}")


def example_export_to_geotiff():
    """Example of exporting data to GeoTIFF format."""
    print("=" * 60)
    print("GEOTIFF EXPORT EXAMPLE")
    print("=" * 60)

    file_path = "path/to/your/pollution_data.nc"

    try:
        with PollutionAnalyzer(file_path, pollution_type="pm25") as analyzer:
            # Export annual average to GeoTIFF
            analyzer.export_to_geotiff(
                output_path="pm25_annual_average.tif",
                time_index=slice("2010-01-01", "2010-12-31"),
                aggregation_method="mean",
            )
            print("Annual average exported to GeoTIFF")

            # Export specific date
            analyzer.export_to_geotiff(
                output_path="pm25_2010_07_15.tif", time_index="2010-07-15"
            )
            print("Specific date exported to GeoTIFF")

            # Export summer average
            analyzer.export_to_geotiff(
                output_path="pm25_summer_2010.tif",
                time_index=slice("2010-06-01", "2010-08-31"),
                aggregation_method="mean",
            )
            print("Summer average exported to GeoTIFF")

    except FileNotFoundError:
        print(f"File not found: {file_path}")


def example_custom_visualization():
    """Custom visualization examples."""
    print("=" * 60)
    print("CUSTOM VISUALIZATION EXAMPLE")
    print("=" * 60)

    file_path = "path/to/your/pollution_data.nc"

    try:
        with PollutionAnalyzer(file_path, pollution_type="bc") as analyzer:
            # Create multiple visualizations

            # 1. Time series at a specific location
            location = {"x": 4321000, "y": 3210000}  # Example LAEA coordinates
            analyzer.plot_time_series(
                location=location,
                title="BC Time Series at Specific Location",
                figsize=(14, 6),
                save_path="bc_timeseries_location.png",
            )

            # 2. Distribution plot
            analyzer.plot_distribution(
                title="BC Distribution Analysis",
                bins=100,
                save_path="bc_distribution.png",
            )

            # 3. Spatial statistics maps
            for stat in ["mean", "max", "std"]:
                analyzer.plot_spatial_statistics(
                    statistic=stat,
                    title=f"BC {stat.title()} Over Time",
                    save_path=f"bc_spatial_{stat}.png",
                )

            # 4. Create animation (if you have enough memory/time)
            # analyzer.create_animation(
            #     output_path="bc_temporal_evolution.gif",
            #     time_step=30,  # Every 30 time steps
            #     fps=2
            # )

            print("Custom visualizations created")

    except FileNotFoundError:
        print(f"File not found: {file_path}")


def example_health_threshold_analysis():
    """Example of health threshold analysis."""
    print("=" * 60)
    print("HEALTH THRESHOLD ANALYSIS EXAMPLE")
    print("=" * 60)

    file_path = "path/to/your/pollution_data.nc"

    try:
        with PollutionAnalyzer(file_path, pollution_type="pm25") as analyzer:
            from pollution_extraction.config import global_config

            # Get health thresholds
            thresholds = global_config.get_health_thresholds("pm25")
            print(f"WHO Annual Guideline: {thresholds['who_annual']} μg/m³")
            print(f"EU Annual Limit: {thresholds['eu_annual']} μg/m³")

            # Calculate annual averages
            analyzer.get_annual_averages()

            # Calculate exceedances
            annual_mean = analyzer.dataset[analyzer.pollution_variable].mean(dim="time")
            who_exceedance = (annual_mean > thresholds["who_annual"]).sum().values
            eu_exceedance = (annual_mean > thresholds["eu_annual"]).sum().values
            total_cells = annual_mean.size

            print("\nExceedance Analysis (based on annual means):")
            print(
                f"  WHO guideline exceeded in {who_exceedance}/{total_cells} grid cells ({100 * who_exceedance / total_cells:.1f}%)"
            )
            print(
                f"  EU limit exceeded in {eu_exceedance}/{total_cells} grid cells ({100 * eu_exceedance / total_cells:.1f}%)"
            )

            # Export annual data for further analysis
            analyzer.export_to_csv(
                "pm25_annual_for_health_analysis.csv", spatial_aggregation="mean"
            )
            print("Data exported for health analysis")

    except FileNotFoundError:
        print(f"File not found: {file_path}")


def example_nuts3_analysis():
    """Example of NUTS3 region analysis for Europe."""
    print("=" * 60)
    print("NUTS3 REGION ANALYSIS EXAMPLE")
    print("=" * 60)

    file_path = "path/to/your/pollution_data.nc"
    nuts3_file = "path/to/nuts3_regions.shp"

    try:
        with PollutionAnalyzer(file_path, pollution_type="no2") as analyzer:
            # Extract data for NUTS3 regions
            nuts3_data = analyzer.extract_for_nuts3(nuts3_file)
            print("Data extracted for NUTS3 regions")

            # Calculate statistics for each region
            nuts3_stats = (
                nuts3_data.groupby("polygon_id")[analyzer.pollution_variable]
                .agg(["mean", "std", "min", "max"])
                .round(2)
            )

            print("NUTS3 Statistics (first 5 regions):")
            print(nuts3_stats.head())

            # Export NUTS3 data to multiple formats
            nuts3_gdf = gpd.read_file(nuts3_file)
            analyzer.exporter.extracted_polygons_to_formats(
                nuts3_data,
                original_polygons=nuts3_gdf,
                output_dir="nuts3_analysis",
                formats=["csv", "geojson", "shapefile"],
                base_filename="no2_nuts3",
            )
            print("NUTS3 analysis exported to multiple formats")

    except FileNotFoundError:
        print("Files not found. Update file paths for your data and NUTS3 shapefile")


def example_batch_processing():
    """Example of batch processing multiple files."""
    print("=" * 60)
    print("BATCH PROCESSING EXAMPLE")
    print("=" * 60)

    # Process multiple files
    file_patterns = ["data_2010.nc", "data_2011.nc", "data_2012.nc"]

    results = {}

    for file_pattern in file_patterns:
        try:
            with PollutionAnalyzer(file_pattern, pollution_type="pm25") as analyzer:
                # Extract year from filename
                year = file_pattern.split("_")[1].split(".")[0]

                # Calculate annual average
                analyzer.get_annual_averages()

                # Export to GeoTIFF
                analyzer.export_to_geotiff(
                    f"pm25_annual_{year}.tif",
                    time_index=slice(None),  # All time steps
                    aggregation_method="mean",
                )

                # Store results
                domain_mean = (
                    analyzer.dataset[analyzer.pollution_variable].mean().values
                )
                results[year] = {
                    "domain_average": float(domain_mean),
                    "file": file_pattern,
                }

                print(
                    f"Processed {file_pattern}: domain average = {domain_mean:.2f} μg/m³"
                )

        except FileNotFoundError:
            print(f"File not found: {file_pattern}")

    # Summary of batch processing
    if results:
        print("\nBatch Processing Summary:")
        for year, data in results.items():
            print(f"  {year}: {data['domain_average']:.2f} μg/m³")


def example_configuration_usage():
    """Example of using custom configuration."""
    print("=" * 60)
    print("CONFIGURATION USAGE EXAMPLE")
    print("=" * 60)

    from pollution_extraction.config import UserConfig

    # Create custom configuration
    config = UserConfig()

    # Modify settings
    config.set("processing", "memory_limit_gb", 16.0)
    config.set("plot_settings", "figure_size", (16, 10))
    config.set("plot_settings", "dpi", 150)

    # Add custom season
    config.add_custom_season("monsoon", [6, 7, 8, 9], "Monsoon Season")

    # Update health guidelines
    config.update_health_guidelines("pm25", custom_threshold=12.0)

    # Save configuration
    config.save("my_custom_config.yaml", format="yaml")
    print("Custom configuration saved to 'my_custom_config.yaml'")

    # Load configuration
    loaded_config = UserConfig("my_custom_config.yaml")
    memory_limit = loaded_config.get("processing", "memory_limit_gb")
    print(f"Loaded memory limit: {memory_limit} GB")

    # Validate configuration
    issues = config.validate_config()
    if not issues:
        print("Configuration is valid")
    else:
        print(f"Configuration issues: {issues}")


def example_quality_control():
    """Example of data quality control checks."""
    print("=" * 60)
    print("QUALITY CONTROL EXAMPLE")
    print("=" * 60)

    file_path = "path/to/your/pollution_data.nc"

    try:
        with PollutionAnalyzer(file_path, pollution_type="pm10") as analyzer:
            from pollution_extraction.utils import (
                calculate_statistics,
                check_data_completeness,
                detect_outliers,
            )

            data = analyzer.data_variable

            # Check data completeness
            completeness = check_data_completeness(data)
            print("Data Completeness Report:")
            print(f"  Total data points: {completeness['total_points']:,}")
            print(f"  Valid data points: {completeness['valid_points']:,}")
            print(f"  Missing data: {completeness['missing_points']:,}")
            print(f"  Coverage: {completeness['coverage_percentage']:.1f}%")
            print(f"  Meets threshold: {completeness['meets_threshold']}")

            # Detect outliers
            data_flat = data.values.flatten()
            valid_data = data_flat[~pd.isna(data_flat)]
            outliers = detect_outliers(valid_data, method="iqr")

            print("\nOutlier Detection:")
            print(f"  Outliers found: {outliers.sum():,}/{len(valid_data):,}")
            print(
                f"  Outlier percentage: {100 * outliers.sum() / len(valid_data):.2f}%"
            )

            # Calculate comprehensive statistics
            stats = calculate_statistics(valid_data)
            print("\nStatistical Summary:")
            print(f"  Mean: {stats['mean']:.2f}")
            print(f"  Std: {stats['std']:.2f}")
            print(f"  Min: {stats['min']:.2f}")
            print(f"  Max: {stats['max']:.2f}")
            print(f"  95th percentile: {stats['p95']:.2f}")

    except FileNotFoundError:
        print(f"File not found: {file_path}")


def example_advanced_spatial_analysis():
    """Advanced spatial analysis example."""
    print("=" * 60)
    print("ADVANCED SPATIAL ANALYSIS EXAMPLE")
    print("=" * 60)

    file_path = "path/to/your/pollution_data.nc"

    try:
        with PollutionAnalyzer(file_path, pollution_type="no2") as analyzer:
            # Extract data with buffer around points
            monitoring_stations = [
                (4321000, 3210000),  # Urban station
                (4500000, 3400000),  # Suburban station
                (4800000, 3600000),  # Rural station
            ]

            # Extract with 5km buffer around each station
            analyzer.spatial_extractor.extract_points(
                monitoring_stations,
                buffer_distance=5000,  # 5 km buffer
            )
            print("Data extracted with 5km buffer around monitoring stations")

            # Spatial subsetting for a specific region
            bounds = {
                "minx": 4200000,
                "maxx": 4600000,
                "miny": 3100000,
                "maxy": 3500000,
            }

            analyzer.spatial_extractor.spatial_subset(bounds)
            print("Data subset for specific geographical bounds")

            # Calculate spatial gradients (urban vs rural)
            urban_data = (
                analyzer.dataset[analyzer.pollution_variable]
                .sel(x=slice(4200000, 4400000), y=slice(3100000, 3300000))
                .mean(dim=["x", "y"])
            )

            rural_data = (
                analyzer.dataset[analyzer.pollution_variable]
                .sel(x=slice(4600000, 4800000), y=slice(3500000, 3700000))
                .mean(dim=["x", "y"])
            )

            urban_mean = float(urban_data.mean())
            rural_mean = float(rural_data.mean())
            gradient = urban_mean - rural_mean

            print("\nUrban-Rural Gradient Analysis:")
            print(f"  Urban average: {urban_mean:.2f} μg/m³")
            print(f"  Rural average: {rural_mean:.2f} μg/m³")
            print(f"  Urban-rural gradient: {gradient:.2f} μg/m³")

    except FileNotFoundError:
        print(f"File not found: {file_path}")


def run_all_examples():
    """Run all examples."""
    examples = [
        example_basic_analysis,
        example_temporal_analysis,
        example_spatial_extraction,
        example_event_based_analysis,
        example_export_to_geotiff,
        example_custom_visualization,
        example_health_threshold_analysis,
        example_nuts3_analysis,
        example_batch_processing,
        example_configuration_usage,
        example_quality_control,
        example_advanced_spatial_analysis,
        example_comprehensive_analysis,
    ]

    print("RUNNING ALL DSS POLLUTION ANALYSIS EXAMPLES")
    print("=" * 80)
    print()

    for i, example_func in enumerate(examples, 1):
        try:
            print(f"Example {i}/{len(examples)}: {example_func.__name__}")
            example_func()
            print("✓ Completed successfully")
        except Exception as e:
            print(f"✗ Error: {e}")

        print()


def create_sample_data():
    """Create sample data for testing examples."""
    print("=" * 60)
    print("CREATING SAMPLE DATA FOR EXAMPLES")
    print("=" * 60)

    import numpy as np
    import xarray as xr

    # Create sample NetCDF file
    print("Creating sample NetCDF file...")

    # Time coordinates
    time = pd.date_range("2010-01-01", "2010-12-31", freq="D")

    # Spatial coordinates (LAEA Europe projection)
    x = np.linspace(4000000, 5000000, 100)
    y = np.linspace(3000000, 4000000, 80)

    # Create realistic pollution patterns
    np.random.seed(42)
    data = np.random.rand(len(time), len(y), len(x)) * 30 + 10  # 10-40 μg/m³

    # Add seasonal variation
    for i, t in enumerate(time):
        seasonal_factor = 1.5 if t.month in [12, 1, 2] else 1.0  # Higher in winter
        data[i, :, :] *= seasonal_factor

    # Create dataset
    ds = xr.Dataset(
        {"PM2p5_downscaled": (["time", "y", "x"], data), "crs": ([], 0)},
        coords={"time": time, "x": x, "y": y},
    )

    # Add attributes
    ds["PM2p5_downscaled"].attrs = {
        "units": "μg/m³",
        "standard_name": "mass_concentration_of_pm2p5_ambient_aerosol_particles_in_air",
        "long_name": "Downscaled PM2.5 concentration in air",
        "grid_mapping": "crs",
    }

    ds.crs.attrs = {
        "grid_mapping_name": "lambert_azimuthal_equal_area",
        "longitude_of_projection_origin": 10.0,
        "latitude_of_projection_origin": 52.0,
        "false_easting": 4321000.0,
        "false_northing": 3210000.0,
        "crs_wkt": 'PROJCS["ETRS89-extended / LAEA Europe",GEOGCS["ETRS89",DATUM["European_Terrestrial_Reference_System_1989",SPHEROID["GRS 1980",6378137,298.257222101]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]],PROJECTION["Lambert_Azimuthal_Equal_Area"],PARAMETER["latitude_of_center",52],PARAMETER["longitude_of_center",10],PARAMETER["false_easting",4321000],PARAMETER["false_northing",3210000],UNIT["metre",1]]',
    }

    # Save sample data
    ds.to_netcdf("sample_pm25_data.nc")
    print("Sample PM2.5 data saved as 'sample_pm25_data.nc'")

    # Create sample points file
    points_df = pd.DataFrame(
        {
            "station_id": ["Urban_Station", "Suburban_Station", "Rural_Station"],
            "x": [4321000, 4500000, 4700000],
            "y": [3210000, 3400000, 3600000],
            "station_type": ["urban", "suburban", "rural"],
        }
    )
    points_df.to_csv("sample_monitoring_stations.csv", index=False)
    print("Sample monitoring stations saved as 'sample_monitoring_stations.csv'")

    print("\nSample data created successfully!")
    print("You can now run the examples using:")
    print("  file_path = 'sample_pm25_data.nc'")


if __name__ == "__main__":
    # Run basic example by default
    # You can also call run_all_examples() to run all examples

    print("DSS Pollution Data Analysis Examples")
    print("To run these examples, please:")
    print("1. Update the file_path variables with your actual NetCDF file paths")
    print("2. Ensure you have the required spatial files (shapefiles, etc.)")
    print("3. Uncomment the example you want to run")
    print()
    print("Or run create_sample_data() first to generate test data")
    print()

    # Uncomment the example you want to run:
    # create_sample_data()
    # example_basic_analysis()
    # example_temporal_analysis()
    # example_spatial_extraction()
    # example_event_based_analysis()
    # example_export_to_geotiff()
    # example_custom_visualization()
    # example_health_threshold_analysis()
    # example_nuts3_analysis()
    # example_batch_processing()
    # example_configuration_usage()
    # example_quality_control()
    # example_advanced_spatial_analysis()
    # example_comprehensive_analysis()
    # run_all_examples()

    print("Please uncomment an example function to run it.")
