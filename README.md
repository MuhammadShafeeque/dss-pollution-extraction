# DSS Pollution Extraction

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Development Status](https://img.shields.io/badge/Development%20Status-Beta-orange.svg)](https://pypi.org/project/dss-pollution-extraction/)

A comprehensive Python package for analyzing pollution data from NetCDF files, developed at the Alfred Wegener Institute (AWI). This package provides tools for temporal aggregations, spatial extractions, visualizations, and health threshold analysis of atmospheric pollution data.

## 🌟 Features

### 📊 **Data Analysis**
- **Multi-pollutant support**: Black Carbon (BC), NO₂, PM₂.₅, PM₁₀
- **Temporal aggregations**: Monthly, seasonal, annual, and custom period averages
- **Spatial extractions**: Point-based, polygon-based, and NUTS3 region analysis
- **Statistical analysis**: Comprehensive data statistics and quality control

### 🗺️ **Visualization**
- **Spatial maps**: Interactive and publication-ready maps with Cartopy support
- **Time series plots**: Domain averages and location-specific trends
- **Seasonal cycles**: Annual pattern analysis and visualization
- **Distribution analysis**: Histograms and box plots for data exploration
- **Spatial statistics**: Mean, maximum, minimum, and standard deviation maps

### 📤 **Data Export**
- **Multiple formats**: NetCDF, GeoTIFF, CSV, GeoJSON, Shapefile
- **Flexible subsetting**: Temporal and spatial data subsetting
- **Batch processing**: Multi-file analysis workflows
- **Compression support**: Optimized file sizes for large datasets

### 🏥 **Health Analysis**
- **WHO guidelines**: Air quality threshold analysis
- **EU standards**: Compliance checking with European regulations
- **Exceedance mapping**: Spatial distribution of threshold violations
- **Health impact assessment**: Tools for public health research

## 🚀 Quick Start

### Installation

```bash
# Install from PyPI (when available)
pip install dss-pollution-extraction

# Or install from source
git clone https://github.com/MuhammadShafeeque/dss-pollution-extraction.git
cd dss-pollution-extraction
pip install -e .
```

### Basic Usage

```python
from pollution_extraction import PollutionAnalyzer

# Initialize analyzer
analyzer = PollutionAnalyzer("your_pollution_data.nc", pollution_type="pm25")

# Print dataset summary
analyzer.print_summary()

# Create visualizations
analyzer.plot_map(time_index=0, save_path="spatial_map.png")
analyzer.plot_time_series(save_path="time_series.png")
analyzer.plot_seasonal_cycle(save_path="seasonal_cycle.png")

# Temporal analysis
monthly_avg = analyzer.get_monthly_averages()
annual_avg = analyzer.get_annual_averages()

# Spatial extraction
point_locations = [(4321000, 3210000), (4500000, 3400000)]
point_data = analyzer.extract_at_points(point_locations)

# Export data
analyzer.export_to_geotiff("pm25_annual.tif", aggregation_method="mean")
analyzer.export_to_csv("pm25_data.csv")
```

## 📁 Project Structure

```
dss-pollution-extraction/
├── pollution_extraction/           # Main package
│   ├── core/                      # Core functionality
│   │   ├── data_reader.py         # NetCDF data reading
│   │   ├── temporal_aggregator.py # Time-based analysis
│   │   ├── spatial_extractor.py   # Spatial data extraction
│   │   ├── data_visualizer.py     # Plotting and visualization
│   │   └── data_exporter.py       # Multi-format data export
│   ├── analyzer.py                # Main analysis interface
│   ├── cli.py                     # Command-line interface
│   ├── config.py                  # Configuration management
│   ├── utils.py                   # Utility functions
│   └── examples.py                # Usage examples
├── examples/                      # Example data and scripts
│   ├── notebooks/                 # Jupyter notebooks
│   ├── scripts/                   # Example Python scripts
│   └── data/                      # Sample datasets
├── tests/                         # Unit tests
├── docs/                          # Documentation
└── config/                        # Configuration files
```

## 🛠️ Requirements

### Core Dependencies
- **Python**: ≥3.8
- **xarray**: ≥2022.3.0 (NetCDF data handling)
- **pandas**: ≥1.4.0 (Data manipulation)
- **numpy**: ≥1.21.0 (Numerical operations)
- **geopandas**: ≥0.10.0 (Spatial data)
- **rioxarray**: ≥0.11.0 (Raster I/O)
- **matplotlib**: ≥3.5.0 (Plotting)
- **seaborn**: ≥0.11.0 (Statistical visualization)
- **cartopy**: ≥0.20.0 (Geographic projections)

### Optional Dependencies
- **Jupyter**: Interactive notebooks
- **Plotly**: Interactive visualizations
- **Folium**: Web-based mapping
- **Numba**: Performance optimization

## 📋 Supported Data Formats

### Input Formats
- **NetCDF4** (.nc): Primary format for atmospheric data
- **Coordinate Systems**: LAEA Europe projection, Geographic (WGS84)
- **Temporal Resolution**: Daily, monthly, annual data
- **Spatial Resolution**: High-resolution gridded data

### Output Formats
- **NetCDF4**: Processed data with metadata preservation
- **GeoTIFF**: Raster format for GIS applications
- **CSV**: Tabular data for statistical analysis
- **GeoJSON**: Vector format for web applications
- **Shapefile**: Standard GIS vector format

## 🌍 Supported Pollutants

| Pollutant | Variable Name | Units | Description |
|-----------|---------------|-------|-------------|
| **Black Carbon (BC)** | `BC_downscaled` | 10⁻⁵ m | Aerosol Optical Depth |
| **Nitrogen Dioxide (NO₂)** | `no2_downscaled` | μg/m³ | Surface concentration |
| **PM₂.₅** | `PM2p5_downscaled` | μg/m³ | Fine particulate matter |
| **PM₁₀** | `PM10_downscaled` | μg/m³ | Coarse particulate matter |

## 📚 Examples

### Temporal Analysis
```python
# Monthly averages for specific years
monthly_data = analyzer.get_monthly_averages(years=[2019, 2020, 2021])

# Seasonal patterns
seasonal_data = analyzer.get_seasonal_averages()

# Custom period analysis
custom_periods = [("2020-03-01", "2020-05-31")]  # COVID lockdown
lockdown_data = analyzer.get_custom_period_averages(custom_periods)
```

### Spatial Extraction
```python
# Extract data for monitoring stations
stations = [(4321000, 3210000), (4500000, 3400000)]
station_data = analyzer.extract_at_points(stations)

# Extract data for administrative regions
nuts3_data = analyzer.extract_for_nuts3("nuts3_regions.shp")

# Extract data for custom polygons
polygon_data = analyzer.extract_for_polygons("study_areas.shp")
```

### Health Threshold Analysis
```python
# WHO and EU guideline analysis
thresholds = analyzer.config.get_health_thresholds("pm25")
annual_avg = analyzer.get_annual_averages()

# Calculate exceedances
who_exceeded = (annual_avg > thresholds["who_annual"]).sum()
eu_exceeded = (annual_avg > thresholds["eu_annual"]).sum()
```

### Batch Processing
```python
# Process multiple files
file_patterns = ["data_2019.nc", "data_2020.nc", "data_2021.nc"]

for file_path in file_patterns:
    with PollutionAnalyzer(file_path, pollution_type="no2") as analyzer:
        # Annual analysis
        annual_avg = analyzer.get_annual_averages()

        # Export results
        year = file_path.split("_")[1].split(".")[0]
        analyzer.export_to_geotiff(f"no2_annual_{year}.tif")
```

## 🖥️ Command Line Interface

The package includes a comprehensive CLI for common operations:

```bash
# Basic dataset information
pollution-cli info pollution_data.nc --type pm25

# Create visualizations
pollution-cli plot pollution_data.nc --type no2 --time-series --spatial-map

# Export data to different formats
pollution-cli export pollution_data.nc --format geotiff --output pm25_annual.tif

# Health threshold analysis
pollution-cli health-analysis pollution_data.nc --type pm25 --who-guidelines

# Batch processing
pollution-cli batch-process data/*.nc --type bc --monthly --annual
```

## 🔧 Configuration

### Default Configuration
The package uses sensible defaults but can be customized via configuration files:

```yaml
# config/user_config.yaml
visualization:
  default_colormap: "viridis"
  figure_size: [12, 8]
  dpi: 300

export:
  compression: "lzw"
  nodata_value: -9999

health_thresholds:
  pm25:
    who_annual: 5.0      # μg/m³
    eu_annual: 25.0      # μg/m³
```

### Environment Variables
```bash
export POLLUTION_CONFIG_PATH="/path/to/your/config.yaml"
export POLLUTION_OUTPUT_DIR="/path/to/output"
```

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=pollution_extraction

# Run specific test categories
pytest tests/test_spatial_extractor.py
```

## 📖 Documentation

Comprehensive documentation is available:

- **API Reference**: Detailed function and class documentation
- **Tutorials**: Step-by-step guides for common use cases
- **Examples**: Jupyter notebooks with real-world applications
- **Installation Guide**: Platform-specific installation instructions

```bash
# Build documentation locally
cd docs/
make html
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details on:

- Setting up the development environment
- Running tests and quality checks
- Submitting pull requests
- Coding standards and guidelines

### Development Setup
```bash
git clone https://github.com/MuhammadShafeeque/dss-pollution-extraction.git
cd dss-pollution-extraction

# Create development environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev,docs,jupyter]"

# Run pre-commit hooks
pre-commit install
```

## 📊 Performance

The package is optimized for large datasets:

- **Chunked processing**: Memory-efficient handling of large NetCDF files
- **Parallel computation**: Multi-core processing with Dask
- **Optimized I/O**: Compressed output formats
- **Caching**: Intelligent caching of intermediate results

### Benchmarks
- **10GB NetCDF file**: ~5 minutes for annual aggregation
- **50 monitoring stations**: <30 seconds for time series extraction
- **NUTS3 regions (1,000+ polygons)**: ~2 minutes for spatial aggregation

## 🐛 Known Issues & Limitations

- **Coordinate Systems**: Currently optimized for LAEA Europe projection
- **Memory Usage**: Very large files (>20GB) may require chunking configuration
- **Projection Support**: Limited support for non-European coordinate systems

See our [Issues](https://github.com/MuhammadShafeeque/dss-pollution-extraction/issues) page for current bugs and feature requests.

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) file for details.

## 👥 Authors & Acknowledgments

**Main Developer:**
- **Muhammad Shafeeque** - *Lead Developer* - [Alfred Wegener Institute (AWI)](https://www.awi.de/)
  - Email: muhammad.shafeeque@awi.de
  - Institution: Alfred Wegener Institute for Polar and Marine Research

**Institution:**
- **Alfred Wegener Institute (AWI)** - Helmholtz Centre for Polar and Marine Research

### Acknowledgments
- European Space Agency (ESA) for satellite data
- Copernicus Atmosphere Monitoring Service (CAMS)
- The xarray and pandas development communities
- Contributors to the atmospheric science Python ecosystem

## 📞 Support & Contact

- **Documentation**: [https://dss-pollution-extraction.readthedocs.io/](https://dss-pollution-extraction.readthedocs.io/)
- **Issues**: [GitHub Issues](https://github.com/MuhammadShafeeque/dss-pollution-extraction/issues)
- **Discussions**: [GitHub Discussions](https://github.com/MuhammadShafeeque/dss-pollution-extraction/discussions)
- **Email**: muhammad.shafeeque@awi.de

## 📈 Roadmap

### Version 1.1 (Planned)
- [ ] Additional coordinate system support
- [ ] Real-time data processing
- [ ] Machine learning integration
- [ ] Performance optimizations

### Version 1.2 (Future)
- [ ] Cloud processing support
- [ ] Additional pollutant types
- [ ] Web-based dashboard
- [ ] API service

## 🏷️ Citation

If you use this package in your research, please cite:

```bibtex
@software{shafeeque2024dss,
  title={DSS Pollution Extraction: A Python Package for Atmospheric Pollution Data Analysis},
  author={Shafeeque, Muhammad},
  year={2024},
  institution={Alfred Wegener Institute},
  url={https://github.com/MuhammadShafeeque/dss-pollution-extraction}
}
```

---

**Keywords:** atmospheric pollution, air quality, NetCDF, spatial analysis, temporal analysis, PM2.5, NO2, black carbon, Python, xarray, environmental data science

For questions, suggestions, or collaborations, please contact the development team at AWI.
