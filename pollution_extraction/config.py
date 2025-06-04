"""Configuration settings for pollution data analysis."""

import logging
from pathlib import Path
from typing import Any, ClassVar


class Config:
    """Configuration class for pollution data analysis."""

    # Default pollution variable mappings
    POLLUTION_VARIABLES: ClassVar[dict[str, dict[str, str]]] = {
        "bc": {
            "var_name": "BC_downscaled",
            "standard_name": (
                "atmosphere_optical_thickness_due_to_black_carbon_"
                "ambient_aerosol_particles"
            ),
            "units": "10^-5 m",
            "description": "Black Carbon Aerosol Optical Depth",
            "colormap": "plasma",
            "data_type": "optical_depth",
        },
        "no2": {
            "var_name": "no2_downscaled",
            "standard_name": "mass_concentration_of_nitrogen_dioxide_in_air",
            "units": "μg/m³",
            "description": "Nitrogen Dioxide Concentration",
            "colormap": "viridis",
            "data_type": "concentration",
        },
        "pm25": {
            "var_name": "PM2p5_downscaled",
            "standard_name": (
                "mass_concentration_of_pm2p5_ambient_aerosol_particles_in_air"
            ),
            "units": "μg/m³",
            "description": "PM2.5 Concentration",
            "colormap": "Reds",
            "data_type": "concentration",
        },
        "pm10": {
            "var_name": "PM10_downscaled",
            "standard_name": (
                "mass_concentration_of_pm10_ambient_aerosol_particles_in_air"
            ),
            "units": "μg/m³",
            "description": "PM10 Concentration",
            "colormap": "Oranges",
            "data_type": "concentration",
        },
    }

    # Season definitions
    SEASONS: ClassVar[dict[str, dict[str, list[int]]]] = {
        "winter": {"months": [12, 1, 2], "name": "Winter (DJF)"},
        "spring": {"months": [3, 4, 5], "name": "Spring (MAM)"},
        "summer": {"months": [6, 7, 8], "name": "Summer (JJA)"},
        "autumn": {"months": [9, 10, 11], "name": "Autumn (SON)"},
        "fall": {"months": [9, 10, 11], "name": "Fall (SON)"},  # Alias
    }

    # Default CRS information for European data
    DEFAULT_CRS: ClassVar[dict[str, Any]] = {
        "epsg": 3035,
        "name": "ETRS89-extended / LAEA Europe",
        "wkt": """PROJCS["ETRS89-extended / LAEA Europe",
                    GEOGCS["ETRS89",
                        DATUM["European_Terrestrial_Reference_System_1989",
                            SPHEROID["GRS 1980",6378137,298.257222101,
                                AUTHORITY["EPSG","7019"]],
                            AUTHORITY["EPSG","6258"]],
                        PRIMEM["Greenwich",0,
                            AUTHORITY["EPSG","8901"]],
                        UNIT["degree",0.0174532925199433,
                            AUTHORITY["EPSG","9122"]],
                        AUTHORITY["EPSG","4258"]],
                    PROJECTION["Lambert_Azimuthal_Equal_Area"],
                    PARAMETER["latitude_of_center",52],
                    PARAMETER["longitude_of_center",10],
                    PARAMETER["false_easting",4321000],
                    PARAMETER["false_northing",3210000],
                    UNIT["metre",1,
                        AUTHORITY["EPSG","9001"]],
                    AUTHORITY["EPSG","3035"]]""",
    }

    # Default data processing settings
    PROCESSING_DEFAULTS: ClassVar[dict[str, Any]] = {
        "chunk_size": {"time": 365, "x": 1000, "y": 1000},
        "memory_limit_gb": 8.0,
        "dask_scheduler": "threads",
        "compression": {"zlib": True, "complevel": 4},
        "interpolation_method": "linear",
        "aggregation_method": "mean",
    }

    # Export format settings
    EXPORT_FORMATS: ClassVar[dict[str, dict[str, Any]]] = {
        "netcdf": {
            "extension": ".nc",
            "compression": {"zlib": True, "complevel": 4},
            "encoding": "utf-8",
        },
        "geotiff": {
            "extension": ".tif",
            "compress": "lzw",
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
            "dtype": "float32",
            "nodata": -9999.0,
        },
        "csv": {
            "extension": ".csv",
            "encoding": "utf-8",
            "index": False,
            "float_format": "%.6f",
        },
        "geojson": {"extension": ".geojson", "driver": "GeoJSON", "encoding": "utf-8"},
        "shapefile": {
            "extension": ".shp",
            "driver": "ESRI Shapefile",
            "encoding": "utf-8",
        },
    }

    # Visualization settings
    PLOT_SETTINGS: ClassVar[dict[str, Any]] = {
        "figure_size": (12, 8),
        "dpi": 300,
        "style": "seaborn-v0_8",
        "color_palette": "husl",
        "font_size": {"title": 14, "label": 12, "tick": 10, "legend": 10},
        "grid_alpha": 0.3,
        "line_width": 2,
        "marker_size": 8,
    }

    # Spatial processing settings
    SPATIAL_SETTINGS: ClassVar[dict[str, Any]] = {
        "buffer_tolerance": 1e-6,
        "simplify_tolerance": 0.0,
        "overlay_method": "intersection",
        "spatial_index": True,
        "validate_geometries": True,
    }

    # Quality control thresholds
    QUALITY_CONTROL: ClassVar[dict[str, Any]] = {
        "missing_data_threshold": 0.5,  # 50% missing data threshold
        "outlier_detection": {"method": "iqr", "threshold": 1.5},
        "temporal_consistency": {"max_gap_days": 30, "min_data_points": 10},
        "spatial_consistency": {
            "neighbor_threshold": 3.0,  # Standard deviations
            "min_neighbors": 4,
        },
    }

    # Logging configuration
    LOGGING_CONFIG: ClassVar[dict[str, Any]] = {
        "level": logging.INFO,
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "datefmt": "%Y-%m-%d %H:%M:%S",
        "handlers": ["console"],  # Can add 'file' handler
    }

    # File naming conventions
    NAMING_CONVENTIONS: ClassVar[dict[str, str]] = {
        "temporal_aggregation": (
            "{pollution_type}_{aggregation}_{period}_{start}_{end}"
        ),
        "spatial_extraction": ("{pollution_type}_extracted_{region_type}_{date}"),
        "time_series": "{pollution_type}_timeseries_{location}_{start}_{end}",
        "statistics": "{pollution_type}_stats_{statistic}_{period}",
        "comparison": "{pollution_type}_comparison_{method}_{date}",
    }

    # Health guidelines and thresholds (WHO, EU standards)
    HEALTH_GUIDELINES: ClassVar[dict[str, dict[str, float]]] = {
        "no2": {
            "who_annual": 10.0,  # μg/m³
            "who_24h": 25.0,  # μg/m³
            "eu_annual": 40.0,  # μg/m³
            "eu_1h": 200.0,  # μg/m³
            "units": "μg/m³",
        },
        "pm25": {
            "who_annual": 5.0,  # μg/m³
            "who_24h": 15.0,  # μg/m³
            "eu_annual": 25.0,  # μg/m³
            "eu_24h": 50.0,  # μg/m³ (not to be exceeded more than 3 times/year)
            "units": "μg/m³",
        },
        "pm10": {
            "who_annual": 15.0,  # μg/m³
            "who_24h": 45.0,  # μg/m³
            "eu_annual": 40.0,  # μg/m³
            "eu_24h": 50.0,  # μg/m³ (not to be exceeded more than 35 times/year)
            "units": "μg/m³",
        },
    }


class UserConfig:
    """User-customizable configuration class."""

    def __init__(self, config_file: str | None = None):
        """Initialize user configuration.

        Parameters
        ----------
        config_file : str, optional
            Path to user configuration file

        """
        self.config_file = config_file
        self._config = {}
        self._load_defaults()

        if config_file:
            self._load_from_file(config_file)

    def _load_defaults(self):
        """Load default configuration."""
        self._config = {
            "pollution_variables": Config.POLLUTION_VARIABLES.copy(),
            "seasons": Config.SEASONS.copy(),
            "processing": Config.PROCESSING_DEFAULTS.copy(),
            "export_formats": Config.EXPORT_FORMATS.copy(),
            "plot_settings": Config.PLOT_SETTINGS.copy(),
            "spatial_settings": Config.SPATIAL_SETTINGS.copy(),
            "quality_control": Config.QUALITY_CONTROL.copy(),
            "logging": Config.LOGGING_CONFIG.copy(),
            "naming": Config.NAMING_CONVENTIONS.copy(),
            "health_guidelines": Config.HEALTH_GUIDELINES.copy(),
        }

    def _load_from_file(self, config_file: str):
        """Load configuration from file."""
        import json

        config_path = Path(config_file)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")

        try:
            # Try YAML first, then JSON
            suffix = config_path.suffix.lower()
            if suffix in [".yml", ".yaml"]:
                try:
                    import yaml
                except ImportError as err:
                    msg = "PyYAML required for YAML files. Run: pip install pyyaml"
                    raise ImportError(msg) from err

                with open(config_path) as f:
                    user_config = yaml.safe_load(f)

            elif suffix == ".json":
                with open(config_path) as f:
                    user_config = json.load(f)

            else:
                raise ValueError(f"Unsupported format: {suffix}")

            # Update configuration with user settings
            self._update_nested_dict(self._config, user_config)

        except Exception as e:
            logging.warning(f"Failed to load config: {e}")

    def _update_nested_dict(self, base_dict: dict, update_dict: dict):
        """Update nested dictionary with new values."""
        for key, value in update_dict.items():
            if (
                key in base_dict
                and isinstance(base_dict[key], dict)
                and isinstance(value, dict)
            ):
                self._update_nested_dict(base_dict[key], value)
            else:
                base_dict[key] = value

    def get(self, section: str, key: str | None = None, default: Any = None) -> Any:
        """Get configuration value.

        Parameters
        ----------
        section : str
            Configuration section
        key : str, optional
            Configuration key within section
        default : any, optional
            Default value if key not found

        Returns
        -------
        any
            Configuration value

        """
        if section not in self._config:
            return default

        if key is None:
            return self._config[section]

        return self._config[section].get(key, default)

    def set(self, section: str, key: str, value: Any):
        """Set configuration value.

        Parameters
        ----------
        section : str
            Configuration section
        key : str
            Configuration key
        value : any
            Configuration value

        """
        if section not in self._config:
            self._config[section] = {}

        self._config[section][key] = value

    def save(self, output_file: str, format: str = "json"):
        """Save configuration to file.

        Parameters
        ----------
        output_file : str
            Output file path
        format : str
            Output format ('json' or 'yaml')

        """
        import json

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Handle different output formats
        if format.lower() == "json":
            with open(output_path, "w") as f:
                json.dump(self._config, f, indent=2)

        elif format.lower() in ["yml", "yaml"]:
            # Try importing yaml and handle missing dependency
            try:
                import yaml
            except ImportError as err:
                msg = "PyYAML required for YAML output. Run: pip install pyyaml"
                raise ImportError(msg) from err

            # Write YAML file
            with open(output_path, "w") as f:
                yaml.safe_dump(self._config, f, indent=2)

        else:
            msg = f"Unsupported format: {format}"
            raise ValueError(msg)

    def get_pollution_config(self, pollution_type: str) -> dict:
        """Get configuration for specific pollution type."""
        return self.get("pollution_variables", pollution_type, {})

    def get_export_config(self, format_name: str) -> dict:
        """Get export configuration for specific format."""
        return self.get("export_formats", format_name, {})

    def get_plot_config(self) -> dict:
        """Get plotting configuration."""
        return self.get("plot_settings")

    def get_health_thresholds(self, pollution_type: str) -> dict:
        """Get health guideline thresholds for pollution type."""
        return self.get("health_guidelines", pollution_type, {})

    def get_processing_config(self) -> dict:
        """Get processing configuration."""
        return self.get("processing")

    def get_spatial_config(self) -> dict:
        """Get spatial processing configuration."""
        return self.get("spatial_settings")

    def get_quality_control_config(self) -> dict:
        """Get quality control configuration."""
        return self.get("quality_control")

    def update_pollution_variable(self, pollution_type: str, **kwargs):
        """Update pollution variable configuration.

        Parameters
        ----------
        pollution_type : str
            Pollution type to update
        **kwargs
            Configuration parameters to update

        """
        if "pollution_variables" not in self._config:
            self._config["pollution_variables"] = {}

        if pollution_type not in self._config["pollution_variables"]:
            self._config["pollution_variables"][pollution_type] = {}

        self._config["pollution_variables"][pollution_type].update(kwargs)

    def add_custom_season(
        self, season_name: str, months: list[int], display_name: str | None = None
    ):
        """Add a custom season definition.

        Parameters
        ----------
        season_name : str
            Name of the custom season
        months : list of int
            List of month numbers (1-12)
        display_name : str, optional
            Display name for the season

        """
        if "seasons" not in self._config:
            self._config["seasons"] = {}

        self._config["seasons"][season_name] = {
            "months": months,
            "name": display_name or season_name.title(),
        }

    def update_health_guidelines(self, pollution_type: str, **thresholds):
        """Update health guidelines for a pollution type.

        Parameters
        ----------
        pollution_type : str
            Pollution type
        **thresholds
            Threshold values to update

        """
        if "health_guidelines" not in self._config:
            self._config["health_guidelines"] = {}

        if pollution_type not in self._config["health_guidelines"]:
            self._config["health_guidelines"][pollution_type] = {}

        self._config["health_guidelines"][pollution_type].update(thresholds)

    def get_all_pollution_types(self) -> list[str]:
        """Get list of all configured pollution types."""
        return list(self._config.get("pollution_variables", {}).keys())

    def get_all_seasons(self) -> list[str]:
        """Get list of all configured seasons."""
        return list(self._config.get("seasons", {}).keys())

    def _validate_pollution_variables(self) -> list[str]:
        """Validate pollution variable configurations."""
        issues = []
        poll_vars = self._config.get("pollution_variables", {})
        for poll_type, config in poll_vars.items():
            required = ["var_name", "standard_name", "units"]
            missing = [key for key in required if key not in config]
            if missing:
                issues.append(f"Missing keys for {poll_type}: {', '.join(missing)}")
        return issues

    def _validate_seasons(self) -> list[str]:
        """Validate season configurations."""
        issues = []
        for name, config in self._config.get("seasons", {}).items():
            if "months" not in config:
                issues.append(f"Missing months for season {name}")
            elif not all(1 <= m <= 12 for m in config["months"]):
                issues.append(f"Invalid month numbers for season {name}")
        return issues

    def _validate_processing(self) -> list[str]:
        """Validate processing settings."""
        issues = []
        proc = self._config.get("processing_defaults", {})

        if "memory_limit_gb" in proc:
            if not isinstance(proc["memory_limit_gb"], int | float):
                issues.append("memory_limit_gb must be numeric")
            elif proc["memory_limit_gb"] <= 0:
                issues.append("memory_limit_gb must be positive")

        if "chunk_size" in proc:
            if not isinstance(proc["chunk_size"], dict):
                issues.append("chunk_size must be a dictionary")

        return issues

    def validate_config(self) -> dict[str, list[str]]:
        """Validate configuration and return any issues."""
        issues = {
            "pollution_variables": self._validate_pollution_variables(),
            "seasons": self._validate_seasons(),
            "processing": self._validate_processing(),
        }
        return {k: v for k, v in issues.items() if v}

    def __str__(self):
        """String representation of configuration."""
        import json

        return json.dumps(self._config, indent=2)


# Global configuration instance
global_config = UserConfig()


def setup_logging(config: UserConfig | None = None):
    """Set up logging based on configuration."""
    if config is None:
        config = global_config

    log_config = config.get("logging")

    logging.basicConfig(
        level=log_config.get("level", logging.INFO),
        format=log_config.get(
            "format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        ),
        datefmt=log_config.get("datefmt", "%Y-%m-%d %H:%M:%S"),
    )


def load_config_from_file(config_file: str) -> UserConfig:
    """Load configuration from file.

    Parameters
    ----------
    config_file : str
        Path to configuration file

    Returns
    -------
    UserConfig
        Loaded configuration instance

    """
    return UserConfig(config_file)


def create_default_config_file(output_path: str, format: str = "yaml"):
    """Create a default configuration file with all available options.

    Parameters
    ----------
    output_path : str
        Output file path
    format : str
        Output format ('yaml' or 'json')

    """
    config = UserConfig()
    config.save(output_path, format=format)
    logging.info(f"Default configuration file created: {output_path}")


def get_config_template() -> dict:
    """Get a configuration template with all available options.

    Returns
    -------
    dict
        Configuration template

    """
    return UserConfig()._config


# Configuration validation functions
def validate_pollution_type(
    pollution_type: str, config: UserConfig | None = None
) -> bool:
    """Validate if pollution type is supported."""
    if config is None:
        config = global_config

    return pollution_type in config.get_all_pollution_types()


def validate_season(season: str, config: UserConfig | None = None) -> bool:
    """Validate if season is defined."""
    if config is None:
        config = global_config

    return season in config.get_all_seasons()


def validate_export_format(format_name: str, config: UserConfig | None = None) -> bool:
    """Validate if export format is supported."""
    if config is None:
        config = global_config

    return format_name in config.get("export_formats", {}).keys()
