"""
Command-line interface for DSS pollution data analysis.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, List
import logging

from .analyzer import PollutionAnalyzer
from .config import setup_logging

logger = logging.getLogger(__name__)


def setup_cli_parser():
    """Set up command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="DSS Pollution Data Extraction and Analysis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis
  dss-pollution-analyze data.nc --type pm25 --info
  
  # Generate time series plot
  dss-pollution-analyze data.nc --type no2 --plot timeseries --output plots/
  
  # Export annual average to GeoTIFF
  dss-pollution-analyze data.nc --type pm25 --export geotiff --time-range 2010-01-01 2010-12-31 --output results/
  
  # Comprehensive analysis
  dss-pollution-analyze data.nc --type bc --comprehensive --regions regions.shp --output analysis/
  
  # Extract data at points
  dss-pollution-analyze data.nc --type no2 --extract points --locations points.csv --output extracted/

  # Monthly aggregation for summer months
  dss-pollution-analyze data.nc --type pm25 --monthly --months 6 7 8 --output summer_analysis/
  
  # Batch processing multiple files
  dss-pollution-analyze data_*.nc --type pm10 --annual --output batch_results/
        """,
    )

    # Required arguments
    parser.add_argument(
        "input_file",
        type=str,
        nargs="+",
        help="Path(s) to input NetCDF file(s). Supports wildcards.",
    )

    # Optional arguments
    parser.add_argument(
        "--type",
        "-t",
        type=str,
        choices=["bc", "no2", "pm25", "pm10"],
        help="Pollution type (auto-detected if not specified)",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="./output",
        help="Output directory (default: ./output)",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    parser.add_argument(
        "--config", type=str, help="Path to configuration file (YAML or JSON)"
    )

    # Analysis options
    analysis_group = parser.add_argument_group("Analysis Options")

    analysis_group.add_argument(
        "--info", action="store_true", help="Display dataset information and statistics"
    )

    analysis_group.add_argument(
        "--comprehensive",
        action="store_true",
        help="Run comprehensive analysis workflow",
    )

    analysis_group.add_argument(
        "--batch", action="store_true", help="Process multiple files in batch mode"
    )

    # Plotting options
    plot_group = parser.add_argument_group("Plotting Options")

    plot_group.add_argument(
        "--plot",
        type=str,
        choices=["timeseries", "map", "seasonal", "distribution", "statistics", "all"],
        help="Generate plots",
    )

    plot_group.add_argument(
        "--time-index",
        type=str,
        help="Time index for spatial plots (date string or integer)",
    )

    plot_group.add_argument(
        "--location",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        help="Coordinates for location-specific time series (x y)",
    )

    # Export options
    export_group = parser.add_argument_group("Export Options")

    export_group.add_argument(
        "--export",
        type=str,
        choices=["netcdf", "geotiff", "csv"],
        help="Export data format",
    )

    export_group.add_argument(
        "--time-range",
        type=str,
        nargs=2,
        metavar=("START", "END"),
        help="Time range for export (start_date end_date)",
    )

    export_group.add_argument(
        "--aggregation",
        type=str,
        choices=["mean", "max", "min", "sum", "std"],
        default="mean",
        help="Temporal aggregation method (default: mean)",
    )

    # Extraction options
    extract_group = parser.add_argument_group("Extraction Options")

    extract_group.add_argument(
        "--extract",
        type=str,
        choices=["points", "polygons", "nuts3"],
        help="Extract data at locations",
    )

    extract_group.add_argument(
        "--locations", type=str, help="Path to locations file (CSV, shapefile, GeoJSON)"
    )

    extract_group.add_argument(
        "--regions", type=str, help="Path to regions file for comprehensive analysis"
    )

    extract_group.add_argument(
        "--points", type=str, help="Path to points file for comprehensive analysis"
    )

    extract_group.add_argument(
        "--buffer", type=float, help="Buffer distance around points (in meters)"
    )

    # Temporal analysis options
    temporal_group = parser.add_argument_group("Temporal Analysis Options")

    temporal_group.add_argument(
        "--monthly", action="store_true", help="Calculate monthly averages"
    )

    temporal_group.add_argument(
        "--annual", action="store_true", help="Calculate annual averages"
    )

    temporal_group.add_argument(
        "--seasonal", action="store_true", help="Calculate seasonal averages"
    )

    temporal_group.add_argument(
        "--months",
        type=int,
        nargs="+",
        metavar="MONTH",
        help="Specific months to include (1-12)",
    )

    temporal_group.add_argument(
        "--years", type=int, nargs="+", metavar="YEAR", help="Specific years to include"
    )

    temporal_group.add_argument(
        "--seasons",
        type=str,
        nargs="+",
        choices=["winter", "spring", "summer", "autumn", "fall"],
        help="Specific seasons to include",
    )

    # Health analysis options
    health_group = parser.add_argument_group("Health Analysis Options")

    health_group.add_argument(
        "--health-analysis",
        action="store_true",
        help="Perform health threshold analysis",
    )

    health_group.add_argument(
        "--who-guidelines",
        action="store_true",
        help="Use WHO guidelines for health analysis",
    )

    health_group.add_argument(
        "--eu-standards",
        action="store_true",
        help="Use EU standards for health analysis",
    )

    # Quality control options
    qc_group = parser.add_argument_group("Quality Control Options")

    qc_group.add_argument(
        "--validate", action="store_true", help="Validate data quality"
    )

    qc_group.add_argument(
        "--outliers", action="store_true", help="Detect and report outliers"
    )

    qc_group.add_argument(
        "--completeness", action="store_true", help="Check data completeness"
    )

    return parser


def handle_info_command(analyzer: PollutionAnalyzer, args: argparse.Namespace):
    """Handle info command."""
    print("\n" + "=" * 80)
    print("DATASET INFORMATION")
    print("=" * 80)

    analyzer.print_summary()

    # Additional statistics if verbose
    if args.verbose:
        info = analyzer.get_info()
        print("\nDETAILED STATISTICS:")
        stats = info["data_summary"]
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"{key.title()}: {value:.6f}")
            else:
                print(f"{key.title()}: {value}")


def handle_plot_command(analyzer: PollutionAnalyzer, args: argparse.Namespace):
    """Handle plot command."""
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    pollution_type = args.type or analyzer.pollution_type

    if args.plot == "timeseries" or args.plot == "all":
        print("Creating time series plot...")
        location = None
        if args.location:
            location = {"x": args.location[0], "y": args.location[1]}

        analyzer.plot_time_series(
            location=location, save_path=plots_dir / f"{pollution_type}_timeseries.png"
        )

    if args.plot == "map" or args.plot == "all":
        print("Creating spatial map...")
        time_idx = 0
        if args.time_index:
            try:
                time_idx = int(args.time_index)
            except ValueError:
                time_idx = args.time_index

        analyzer.plot_map(
            time_index=time_idx, save_path=plots_dir / f"{pollution_type}_map.png"
        )

    if args.plot == "seasonal" or args.plot == "all":
        print("Creating seasonal cycle plot...")
        analyzer.plot_seasonal_cycle(
            save_path=plots_dir / f"{pollution_type}_seasonal.png"
        )

    if args.plot == "distribution" or args.plot == "all":
        print("Creating distribution plot...")
        analyzer.plot_distribution(
            save_path=plots_dir / f"{pollution_type}_distribution.png"
        )

    if args.plot == "statistics" or args.plot == "all":
        print("Creating spatial statistics plots...")
        for stat in ["mean", "max", "std"]:
            analyzer.plot_spatial_statistics(
                statistic=stat,
                save_path=plots_dir / f"{pollution_type}_spatial_{stat}.png",
            )

    print(f"Plots saved to: {plots_dir}")


def handle_export_command(analyzer: PollutionAnalyzer, args: argparse.Namespace):
    """Handle export command."""
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    exports_dir = output_dir / "exports"
    exports_dir.mkdir(exist_ok=True)

    pollution_type = args.type or analyzer.pollution_type

    # Determine time range
    time_slice = None
    time_suffix = "all"

    if args.time_range:
        start_date, end_date = args.time_range
        time_slice = slice(start_date, end_date)
        time_suffix = f"{start_date}_{end_date}"

    if args.export == "netcdf":
        output_file = exports_dir / f"{pollution_type}_{time_suffix}.nc"
        print(f"Exporting to NetCDF: {output_file}")
        analyzer.export_to_netcdf(output_path=output_file, time_subset=time_slice)

    elif args.export == "geotiff":
        output_file = (
            exports_dir / f"{pollution_type}_{args.aggregation}_{time_suffix}.tif"
        )
        print(f"Exporting to GeoTIFF: {output_file}")
        analyzer.export_to_geotiff(
            output_path=output_file,
            time_index=time_slice or slice(None),
            aggregation_method=args.aggregation,
        )

    elif args.export == "csv":
        output_file = exports_dir / f"{pollution_type}_{time_suffix}.csv"
        print(f"Exporting to CSV: {output_file}")
        analyzer.export_to_csv(
            output_path=output_file, spatial_aggregation=args.aggregation
        )

    print(f"Export completed: {output_file}")


def handle_extract_command(analyzer: PollutionAnalyzer, args: argparse.Namespace):
    """Handle extract command."""
    if not args.locations:
        print("Error: --locations file required for extraction")
        return

    output_dir = Path(args.output)
    extract_dir = output_dir / "extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)

    pollution_type = args.type or analyzer.pollution_type

    if args.extract == "points":
        print(f"Extracting data at points from: {args.locations}")
        extracted_data = analyzer.extract_at_points(
            args.locations, buffer_distance=args.buffer
        )

        analyzer.exporter.extracted_points_to_formats(
            extracted_data,
            output_dir=extract_dir,
            formats=["csv", "geojson"],
            base_filename=f"{pollution_type}_points",
        )

    elif args.extract == "polygons":
        print(f"Extracting data for polygons from: {args.locations}")

        # Load polygons
        import geopandas as gpd

        polygons_gdf = gpd.read_file(args.locations)

        extracted_data = analyzer.extract_for_polygons(polygons_gdf)

        analyzer.exporter.extracted_polygons_to_formats(
            extracted_data,
            original_polygons=polygons_gdf,
            output_dir=extract_dir,
            formats=["csv", "geojson"],
            base_filename=f"{pollution_type}_polygons",
        )

    elif args.extract == "nuts3":
        print(f"Extracting data for NUTS3 regions from: {args.locations}")
        extracted_data = analyzer.extract_for_nuts3(args.locations)

        # Load NUTS3 for export
        import geopandas as gpd

        nuts3_gdf = gpd.read_file(args.locations)

        analyzer.exporter.extracted_polygons_to_formats(
            extracted_data,
            original_polygons=nuts3_gdf,
            output_dir=extract_dir,
            formats=["csv", "geojson"],
            base_filename=f"{pollution_type}_nuts3",
        )

    print(f"Extraction completed. Results saved to: {extract_dir}")


def handle_temporal_analysis(analyzer: PollutionAnalyzer, args: argparse.Namespace):
    """Handle temporal analysis commands."""
    output_dir = Path(args.output)
    temporal_dir = output_dir / "temporal"
    temporal_dir.mkdir(parents=True, exist_ok=True)

    pollution_type = args.type or analyzer.pollution_type

    if args.monthly:
        print("Calculating monthly averages...")
        monthly_data = analyzer.get_monthly_averages(
            months=args.months, years=args.years
        )
        monthly_data.to_netcdf(temporal_dir / f"{pollution_type}_monthly.nc")
        print(f"Monthly data saved to: {temporal_dir}")

    if args.annual:
        print("Calculating annual averages...")
        annual_data = analyzer.get_annual_averages(years=args.years)
        annual_data.to_netcdf(temporal_dir / f"{pollution_type}_annual.nc")
        print(f"Annual data saved to: {temporal_dir}")

    if args.seasonal:
        print("Calculating seasonal averages...")
        seasonal_data = analyzer.get_seasonal_averages(
            seasons=args.seasons, years=args.years
        )
        seasonal_data.to_netcdf(temporal_dir / f"{pollution_type}_seasonal.nc")
        print(f"Seasonal data saved to: {temporal_dir}")


def handle_health_analysis(analyzer: PollutionAnalyzer, args: argparse.Namespace):
    """Handle health analysis commands."""
    from .config import global_config

    output_dir = Path(args.output)
    health_dir = output_dir / "health_analysis"
    health_dir.mkdir(parents=True, exist_ok=True)

    pollution_type = args.type or analyzer.pollution_type

    # Get health thresholds
    thresholds = global_config.get_health_thresholds(pollution_type)

    if not thresholds:
        print(f"No health guidelines available for {pollution_type}")
        return

    print(f"\n{pollution_type.upper()} Health Guidelines:")

    if args.who_guidelines or not args.eu_standards:
        if "who_annual" in thresholds:
            print(f"  WHO Annual: {thresholds['who_annual']} {thresholds['units']}")
        if "who_24h" in thresholds:
            print(f"  WHO 24-hour: {thresholds['who_24h']} {thresholds['units']}")

    if args.eu_standards or not args.who_guidelines:
        if "eu_annual" in thresholds:
            print(f"  EU Annual: {thresholds['eu_annual']} {thresholds['units']}")
        if "eu_24h" in thresholds:
            print(f"  EU 24-hour: {thresholds['eu_24h']} {thresholds['units']}")

    # Calculate annual mean for exceedance analysis
    annual_mean = analyzer.dataset[analyzer.pollution_variable].mean(dim="time")

    # WHO exceedances
    if "who_annual" in thresholds:
        who_exceed = (annual_mean > thresholds["who_annual"]).sum().values
        total_cells = annual_mean.size
        print(f"\nWHO Guideline Exceedances:")
        print(
            f"  {who_exceed}/{total_cells} grid cells ({100 * who_exceed / total_cells:.1f}%)"
        )

    # EU exceedances
    if "eu_annual" in thresholds:
        eu_exceed = (annual_mean > thresholds["eu_annual"]).sum().values
        total_cells = annual_mean.size
        print(f"EU Limit Exceedances:")
        print(
            f"  {eu_exceed}/{total_cells} grid cells ({100 * eu_exceed / total_cells:.1f}%)"
        )


def handle_quality_control(analyzer: PollutionAnalyzer, args: argparse.Namespace):
    """Handle quality control commands."""
    from .utils import check_data_completeness, detect_outliers

    print("\nQUALITY CONTROL ANALYSIS")
    print("=" * 50)

    data = analyzer.data_variable

    if args.completeness:
        print("Checking data completeness...")
        completeness = check_data_completeness(data)
        print(f"  Data Coverage: {completeness['coverage_percentage']:.1f}%")
        print(f"  Missing Points: {completeness['missing_points']:,}")
        print(f"  Meets Threshold: {completeness['meets_threshold']}")

    if args.outliers:
        print("Detecting outliers...")
        data_values = data.values.flatten()
        outliers = detect_outliers(data_values[~pd.isna(data_values)])
        outlier_count = outliers.sum()
        total_count = len(data_values[~pd.isna(data_values)])
        print(
            f"  Outliers Found: {outlier_count:,}/{total_count:,} ({100 * outlier_count / total_count:.2f}%)"
        )


def handle_comprehensive_analysis(
    analyzer: PollutionAnalyzer, args: argparse.Namespace
):
    """Handle comprehensive analysis command."""
    output_dir = Path(args.output)

    print("Running comprehensive analysis...")
    print(f"Output directory: {output_dir}")

    results = analyzer.comprehensive_analysis(
        output_dir=output_dir, regions_file=args.regions, points_file=args.points
    )

    print("Comprehensive analysis completed!")
    print(f"Results available in: {output_dir}")
    print(f"Analysis components: {list(results.keys())}")


def process_single_file(file_path: Path, args: argparse.Namespace):
    """Process a single NetCDF file."""
    print(f"\nProcessing: {file_path}")

    try:
        with PollutionAnalyzer(file_path, pollution_type=args.type) as analyzer:
            print(f"Loaded {analyzer.pollution_type.upper()} data")

            # Handle different commands
            if args.info:
                handle_info_command(analyzer, args)

            if args.plot:
                handle_plot_command(analyzer, args)

            if args.export:
                handle_export_command(analyzer, args)

            if args.extract:
                handle_extract_command(analyzer, args)

            # Temporal analysis
            if args.monthly or args.annual or args.seasonal:
                handle_temporal_analysis(analyzer, args)

            # Health analysis
            if args.health_analysis or args.who_guidelines or args.eu_standards:
                handle_health_analysis(analyzer, args)

            # Quality control
            if args.validate or args.outliers or args.completeness:
                handle_quality_control(analyzer, args)

            if args.comprehensive:
                handle_comprehensive_analysis(analyzer, args)

            # If no specific command, show info by default
            if not any(
                [
                    args.info,
                    args.plot,
                    args.export,
                    args.extract,
                    args.monthly,
                    args.annual,
                    args.seasonal,
                    args.comprehensive,
                    args.health_analysis,
                    args.validate,
                ]
            ):
                print("No specific command specified. Showing dataset info:")
                handle_info_command(analyzer, args)

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return False

    return True


def main():
    """Main CLI entry point."""
    parser = setup_cli_parser()
    args = parser.parse_args()

    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Load configuration if provided
    if args.config:
        from .config import UserConfig

        try:
            config = UserConfig(args.config)
            print(f"Loaded configuration from: {args.config}")
        except Exception as e:
            print(f"Warning: Could not load configuration: {e}")

    # Validate input files
    input_files = []
    for pattern in args.input_file:
        files = list(Path().glob(pattern))
        if files:
            input_files.extend(files)
        else:
            file_path = Path(pattern)
            if file_path.exists():
                input_files.append(file_path)
            else:
                print(f"Warning: File not found: {pattern}")

    if not input_files:
        print("Error: No valid input files found")
        sys.exit(1)

    print(f"Found {len(input_files)} file(s) to process")

    # Process files
    successful = 0
    failed = 0

    for file_path in input_files:
        if args.batch and len(input_files) > 1:
            # Create separate output directory for each file in batch mode
            file_output_dir = Path(args.output) / file_path.stem
            args.output = str(file_output_dir)

        if process_single_file(file_path, args):
            successful += 1
        else:
            failed += 1

    # Summary
    print(f"\n{'=' * 60}")
    print("PROCESSING SUMMARY")
    print(f"{'=' * 60}")
    print(f"Files processed successfully: {successful}")
    print(f"Files failed: {failed}")
    print(f"Output directory: {args.output}")

    if failed > 0:
        print("Some files failed to process. Check the error messages above.")
        sys.exit(1)

    print("Analysis completed successfully!")


if __name__ == "__main__":
    main()
