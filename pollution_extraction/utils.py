"""Utility functions for pollution data analysis."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)


def validate_date_range(
    start_date: str, end_date: str
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Validate and parse date range.

    Parameters
    ----------
    start_date : str
        Start date string
    end_date : str
        End date string

    Returns:
    -------
    tuple
        Tuple of parsed start and end dates

    Raises:
    ------
    ValueError
        If dates are invalid or start_date > end_date
    """
    try:
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
    except Exception as e:
        raise ValueError(f"Invalid date format: {e}")

    if start > end:
        raise ValueError(
            f"Start date ({start_date}) must be before end date ({end_date})"
        )

    return start, end


def validate_coordinates(
    x: float,
    y: float,
    x_bounds: tuple[float, float] | None = None,
    y_bounds: tuple[float, float] | None = None,
) -> bool:
    """Validate coordinate values.

    Parameters
    ----------
    x, y : float
        Coordinate values
    x_bounds, y_bounds : tuple, optional
        Valid coordinate bounds

    Returns:
    -------
    bool
        True if coordinates are valid
    """
    if not (np.isfinite(x) and np.isfinite(y)):
        return False

    if x_bounds and not (x_bounds[0] <= x <= x_bounds[1]):
        return False

    if y_bounds and not (y_bounds[0] <= y <= y_bounds[1]):
        return False

    return True


def get_season_from_month(month: int) -> str:
    """Get season from month number.

    Parameters
    ----------
    month : int
        Month number (1-12)

    Returns:
    -------
    str
        Season name
    """
    season_map = {
        12: "winter",
        1: "winter",
        2: "winter",
        3: "spring",
        4: "spring",
        5: "spring",
        6: "summer",
        7: "summer",
        8: "summer",
        9: "autumn",
        10: "autumn",
        11: "autumn",
    }
    return season_map.get(month, "unknown")


def calculate_statistics(
    data: np.ndarray, percentiles: list[float] | None = None
) -> dict[str, float]:
    """Calculate comprehensive statistics for data array.

    Parameters
    ----------
    data : np.ndarray
        Input data array
    percentiles : list of float
        Percentiles to calculate

    Returns:
    -------
    dict
        Dictionary of statistics
    """
    # Remove NaN values
    if percentiles is None:
        percentiles = [25, 50, 75, 90, 95, 99]
    clean_data = data[~np.isnan(data)]

    if len(clean_data) == 0:
        return {
            stat: np.nan
            for stat in ["mean", "std", "min", "max", "count"]
            + [f"p{p}" for p in percentiles]
        }

    stats = {
        "count": len(clean_data),
        "mean": np.mean(clean_data),
        "std": np.std(clean_data),
        "min": np.min(clean_data),
        "max": np.max(clean_data),
        "skew": float(pd.Series(clean_data).skew()),
        "kurtosis": float(pd.Series(clean_data).kurtosis()),
    }

    # Add percentiles
    for p in percentiles:
        stats[f"p{p}"] = np.percentile(clean_data, p)

    return stats


def detect_outliers(
    data: np.ndarray, method: str = "iqr", threshold: float = 1.5
) -> np.ndarray:
    """Detect outliers in data using various methods.

    Parameters
    ----------
    data : np.ndarray
        Input data
    method : str
        Outlier detection method ('iqr', 'zscore', 'modified_zscore')
    threshold : float
        Threshold for outlier detection

    Returns:
    -------
    np.ndarray
        Boolean array indicating outliers
    """
    clean_data = data[~np.isnan(data)]

    if len(clean_data) == 0:
        return np.zeros_like(data, dtype=bool)

    if method == "iqr":
        q1 = np.percentile(clean_data, 25)
        q3 = np.percentile(clean_data, 75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        outliers = (data < lower_bound) | (data > upper_bound)

    elif method == "zscore":
        z_scores = np.abs((data - np.nanmean(data)) / np.nanstd(data))
        outliers = z_scores > threshold

    elif method == "modified_zscore":
        median = np.nanmedian(data)
        mad = np.nanmedian(np.abs(data - median))
        modified_z_scores = 0.6745 * (data - median) / mad
        outliers = np.abs(modified_z_scores) > threshold

    else:
        raise ValueError(f"Unknown outlier detection method: {method}")

    return outliers


def resample_data(
    data: xr.Dataset, target_resolution: dict[str, float], method: str = "linear"
) -> xr.Dataset:
    """Resample spatial data to target resolution.

    Parameters
    ----------
    data : xr.Dataset
        Input dataset
    target_resolution : dict
        Target resolution with 'x' and 'y' keys
    method : str
        Interpolation method

    Returns:
    -------
    xr.Dataset
        Resampled dataset
    """
    # Calculate new coordinate arrays
    x_min, x_max = float(data.x.min()), float(data.x.max())
    y_min, y_max = float(data.y.min()), float(data.y.max())

    new_x = np.arange(x_min, x_max + target_resolution["x"], target_resolution["x"])
    new_y = np.arange(y_min, y_max + target_resolution["y"], target_resolution["y"])

    # Interpolate to new grid
    resampled = data.interp(x=new_x, y=new_y, method=method)

    return resampled


def create_time_mask(
    time_coord: xr.DataArray,
    include_months: list[int] | None = None,
    include_seasons: list[str] | None = None,
    include_years: list[int] | None = None,
    exclude_weekends: bool = False,
) -> xr.DataArray:
    """Create a time mask for filtering data.

    Parameters
    ----------
    time_coord : xr.DataArray
        Time coordinate array
    include_months : list of int, optional
        Months to include (1-12)
    include_seasons : list of str, optional
        Seasons to include
    include_years : list of int, optional
        Years to include
    exclude_weekends : bool
        Whether to exclude weekends

    Returns:
    -------
    xr.DataArray
        Boolean mask array
    """
    mask = xr.ones_like(time_coord, dtype=bool)

    if include_months:
        month_mask = time_coord.dt.month.isin(include_months)
        mask = mask & month_mask

    if include_seasons:
        season_months = []
        season_map = {
            "winter": [12, 1, 2],
            "spring": [3, 4, 5],
            "summer": [6, 7, 8],
            "autumn": [9, 10, 11],
            "fall": [9, 10, 11],
        }

        for season in include_seasons:
            if season in season_map:
                season_months.extend(season_map[season])

        if season_months:
            season_mask = time_coord.dt.month.isin(season_months)
            mask = mask & season_mask

    if include_years:
        year_mask = time_coord.dt.year.isin(include_years)
        mask = mask & year_mask

    if exclude_weekends:
        weekday_mask = time_coord.dt.weekday < 5  # Monday=0, Sunday=6
        mask = mask & weekday_mask

    return mask


def align_datasets(
    datasets: list[xr.Dataset], method: str = "inner"
) -> list[xr.Dataset]:
    """Align multiple datasets to common coordinates.

    Parameters
    ----------
    datasets : list of xr.Dataset
        List of datasets to align
    method : str
        Alignment method ('inner', 'outer', 'left', 'right')

    Returns:
    -------
    list of xr.Dataset
        Aligned datasets
    """
    if len(datasets) < 2:
        return datasets

    # Find common coordinates
    ref_dataset = datasets[0]

    aligned_datasets = []
    for dataset in datasets:
        # Align with reference dataset
        aligned = xr.align(ref_dataset, dataset, join=method)[1]
        aligned_datasets.append(aligned)

    return aligned_datasets


def calculate_trends(
    data: xr.DataArray, time_dim: str = "time", method: str = "linear"
) -> xr.Dataset:
    """Calculate temporal trends in data.

    Parameters
    ----------
    data : xr.DataArray
        Input data array
    time_dim : str
        Name of time dimension
    method : str
        Trend calculation method ('linear', 'theil_sen')

    Returns:
    -------
    xr.Dataset
        Dataset with trend statistics (slope, intercept, r_value, p_value)
    """
    from scipy import stats

    def calculate_trend_stats(y):
        """Calculate trend statistics for 1D array."""
        if np.isnan(y).all():
            return np.nan, np.nan, np.nan, np.nan

        x = np.arange(len(y))
        valid_mask = ~np.isnan(y)

        if np.sum(valid_mask) < 3:
            return np.nan, np.nan, np.nan, np.nan

        if method == "linear":
            slope, intercept, r_value, p_value, _ = stats.linregress(
                x[valid_mask], y[valid_mask]
            )
        elif method == "theil_sen":
            slope, intercept, _, _ = stats.theilslopes(y[valid_mask], x[valid_mask])
            r_value, p_value = np.nan, np.nan  # Theil-Sen doesn't provide these
        else:
            raise ValueError(f"Unknown trend method: {method}")

        return slope, intercept, r_value, p_value

    # Apply trend calculation
    result = xr.apply_ufunc(
        calculate_trend_stats,
        data,
        input_core_dims=[[time_dim]],
        output_core_dims=[[], [], [], []],
        dask="allowed",
        output_dtypes=[float, float, float, float],
        vectorize=True,
    )

    # Create dataset with results
    trend_ds = xr.Dataset(
        {
            "slope": (data.dims[1:], result[0]),
            "intercept": (data.dims[1:], result[1]),
            "r_value": (data.dims[1:], result[2]),
            "p_value": (data.dims[1:], result[3]),
        }
    )

    # Copy coordinates (excluding time)
    for coord in data.coords:
        if coord != time_dim:
            trend_ds.coords[coord] = data.coords[coord]

    return trend_ds


def convert_units(data: xr.DataArray, from_unit: str, to_unit: str) -> xr.DataArray:
    """Convert data units.

    Parameters
    ----------
    data : xr.DataArray
        Input data
    from_unit : str
        Source unit
    to_unit : str
        Target unit

    Returns:
    -------
    xr.DataArray
        Data with converted units
    """
    # Unit conversion factors (basic implementation)
    conversion_factors = {
        ("μg/m³", "mg/m³"): 0.001,
        ("mg/m³", "μg/m³"): 1000,
        ("μg/m³", "ng/m³"): 1000,
        ("ng/m³", "μg/m³"): 0.001,
        ("10^-5 m", "m"): 1e-5,
        ("m", "10^-5 m"): 1e5,
    }

    key = (from_unit, to_unit)

    if key in conversion_factors:
        converted_data = data * conversion_factors[key]
        converted_data.attrs = data.attrs.copy()
        converted_data.attrs["units"] = to_unit
        return converted_data
    elif from_unit == to_unit:
        return data
    else:
        logger.warning(f"No conversion available from {from_unit} to {to_unit}")
        return data


def memory_usage_check(
    dataset: xr.Dataset, operation: str = "load", memory_limit_gb: float = 8.0
) -> bool:
    """Check if dataset operation will exceed memory limits.

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset to check
    operation : str
        Type of operation ('load', 'compute', 'plot')
    memory_limit_gb : float
        Memory limit in GB

    Returns:
    -------
    bool
        True if operation is safe
    """
    # Estimate memory usage
    total_size = 0
    for var in dataset.data_vars:
        var_size = dataset[var].nbytes
        total_size += var_size

    # Add overhead for different operations
    overhead_factors = {"load": 1.2, "compute": 2.0, "plot": 1.5}

    estimated_memory_gb = total_size * overhead_factors.get(operation, 1.5) / 1e9

    if estimated_memory_gb > memory_limit_gb:
        logger.warning(
            f"Operation '{operation}' may require {estimated_memory_gb:.2f} GB "
            f"(limit: {memory_limit_gb:.2f} GB). Consider using dask or subsetting data."
        )
        return False

    return True


def optimize_chunks(
    dataset: xr.Dataset, target_chunk_size_mb: float = 128
) -> xr.Dataset:
    """Optimize chunk sizes for dask operations.

    Parameters
    ----------
    dataset : xr.Dataset
        Input dataset
    target_chunk_size_mb : float
        Target chunk size in MB

    Returns:
    -------
    xr.Dataset
        Dataset with optimized chunks
    """
    # Calculate optimal chunk sizes
    target_size_bytes = target_chunk_size_mb * 1e6

    chunks = {}
    for dim in dataset.dims:
        dim_size = dataset.dims[dim]

        if dim == "time":
            # For time dimension, use smaller chunks for temporal operations
            chunks[dim] = min(365, dim_size)  # Annual chunks or less
        else:
            # For spatial dimensions
            element_size = 4  # Assume float32
            chunk_elements = int(target_size_bytes / element_size)
            chunk_size = int(np.sqrt(chunk_elements))  # Square chunks
            chunks[dim] = min(chunk_size, dim_size)

    return dataset.chunk(chunks)


def validate_crs(dataset: xr.Dataset) -> bool:
    """Validate coordinate reference system information.

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset to validate

    Returns:
    -------
    bool
        True if CRS is valid
    """
    if "crs" not in dataset:
        logger.warning("No CRS information found in dataset")
        return False

    crs_attrs = dataset.crs.attrs
    required_attrs = ["grid_mapping_name"]

    for attr in required_attrs:
        if attr not in crs_attrs:
            logger.warning(f"Missing required CRS attribute: {attr}")
            return False

    return True


def create_directory_structure(
    base_dir: Path, analysis_type: str = "comprehensive"
) -> dict[str, Path]:
    """Create standardized directory structure for analysis outputs.

    Parameters
    ----------
    base_dir : Path
        Base output directory
    analysis_type : str
        Type of analysis ('temporal', 'spatial', 'comprehensive')

    Returns:
    -------
    dict
        Dictionary of created directories
    """
    base_dir = Path(base_dir)

    directories = {
        "base": base_dir,
        "data": base_dir / "data",
        "plots": base_dir / "plots",
        "exports": base_dir / "exports",
        "metadata": base_dir / "metadata",
    }

    if analysis_type in ["temporal", "comprehensive"]:
        directories.update(
            {
                "temporal_data": directories["data"] / "temporal",
                "temporal_plots": directories["plots"] / "temporal",
            }
        )

    if analysis_type in ["spatial", "comprehensive"]:
        directories.update(
            {
                "spatial_data": directories["data"] / "spatial",
                "spatial_plots": directories["plots"] / "spatial",
            }
        )

    # Create all directories
    for dir_path in directories.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    return directories


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format.

    Parameters
    ----------
    size_bytes : int
        Size in bytes

    Returns:
    -------
    str
        Formatted size string
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def estimate_processing_time(dataset_size_gb: float, operation: str = "load") -> str:
    """Estimate processing time based on dataset size.

    Parameters
    ----------
    dataset_size_gb : float
        Dataset size in GB
    operation : str
        Type of operation ('load', 'aggregate', 'extract', 'export')

    Returns:
    -------
    str
        Estimated time string
    """
    # Rough estimates based on typical hardware
    time_factors = {
        "load": 0.5,  # minutes per GB
        "aggregate": 0.2,  # minutes per GB
        "extract": 0.3,  # minutes per GB
        "export": 0.4,  # minutes per GB
    }

    factor = time_factors.get(operation, 0.3)
    estimated_minutes = dataset_size_gb * factor

    if estimated_minutes < 1:
        return f"{estimated_minutes * 60:.0f} seconds"
    elif estimated_minutes < 60:
        return f"{estimated_minutes:.1f} minutes"
    else:
        hours = estimated_minutes / 60
        return f"{hours:.1f} hours"


def check_data_completeness(
    data: xr.DataArray, min_coverage: float = 0.8
) -> dict[str, Any]:
    """Check data completeness and identify gaps.

    Parameters
    ----------
    data : xr.DataArray
        Input data array
    min_coverage : float
        Minimum data coverage threshold (0-1)

    Returns:
    -------
    dict
        Completeness report
    """
    total_points = data.size
    valid_points = (~data.isnull()).sum().item()
    coverage = valid_points / total_points

    report = {
        "total_points": total_points,
        "valid_points": valid_points,
        "missing_points": total_points - valid_points,
        "coverage_ratio": coverage,
        "coverage_percentage": coverage * 100,
        "meets_threshold": coverage >= min_coverage,
        "threshold": min_coverage,
    }

    # Check temporal gaps
    if "time" in data.dims:
        time_coverage = (~data.isnull()).sum(dim=["x", "y"]) / (
            data.sizes["x"] * data.sizes["y"]
        )
        report["temporal_coverage"] = {
            "mean": float(time_coverage.mean()),
            "min": float(time_coverage.min()),
            "max": float(time_coverage.max()),
        }

    # Check spatial gaps
    if "x" in data.dims and "y" in data.dims:
        spatial_coverage = (~data.isnull()).sum(dim="time") / data.sizes["time"]
        report["spatial_coverage"] = {
            "mean": float(spatial_coverage.mean()),
            "min": float(spatial_coverage.min()),
            "max": float(spatial_coverage.max()),
        }

    return report
