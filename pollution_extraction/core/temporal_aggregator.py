"""Temporal aggregation module for pollution data."""

import logging

import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)


class TemporalAggregator:
    """Class for performing temporal aggregations on pollution data.

    Supports various aggregation methods including mean, sum, min, max, std.
    """

    # Season definitions
    SEASONS = {
        "winter": [12, 1, 2],
        "spring": [3, 4, 5],
        "summer": [6, 7, 8],
        "autumn": [9, 10, 11],
        "fall": [9, 10, 11],  # Alias for autumn
    }

    # Available aggregation methods
    AGGREGATION_METHODS = ["mean", "sum", "min", "max", "std", "median", "count"]

    def __init__(self, dataset: xr.Dataset, pollution_variable: str):
        """Initialize the temporal aggregator.

        Parameters
        ----------
        dataset : xr.Dataset
            The pollution dataset
        pollution_variable : str
            Name of the pollution variable to aggregate
        """
        self.dataset = dataset
        self.pollution_variable = pollution_variable

    def _validate_method(self, method: str) -> None:
        """Validate aggregation method."""
        if method not in self.AGGREGATION_METHODS:
            raise ValueError(
                f"Method '{method}' not supported. "
                f"Available methods: {self.AGGREGATION_METHODS}"
            )

    def _add_time_coordinates(self, data: xr.Dataset) -> xr.Dataset:
        """Add useful time coordinates for grouping operations."""
        data = data.copy()
        data.coords["year"] = data.time.dt.year
        data.coords["month"] = data.time.dt.month
        data.coords["day"] = data.time.dt.day
        data.coords["dayofyear"] = data.time.dt.dayofyear
        data.coords["season"] = ("time", self._get_seasons(data.time))
        return data

    def _get_seasons(self, time_coord: xr.DataArray) -> list[str]:
        """Get season labels for each time point."""
        months = time_coord.dt.month.values
        seasons = []

        for month in months:
            for season_name, season_months in self.SEASONS.items():
                if season_name != "fall" and month in season_months:
                    seasons.append(season_name)
                    break

        return seasons

    def daily_aggregation(self, method: str = "mean") -> xr.Dataset:
        """Aggregate to daily values (useful if data has sub-daily resolution).

        Parameters
        ----------
        method : str
            Aggregation method

        Returns:
        -------
        xr.Dataset
            Daily aggregated dataset
        """
        self._validate_method(method)

        data_with_coords = self._add_time_coordinates(self.dataset)

        if method == "count":
            result = data_with_coords.groupby("time.date").count()
        else:
            agg_func = getattr(data_with_coords.groupby("time.date"), method)
            result = agg_func()

        return result

    def monthly_aggregation(
        self,
        method: str = "mean",
        specific_months: list[int] | None = None,
        years: list[int] | None = None,
    ) -> xr.Dataset:
        """Aggregate to monthly values.

        Parameters
        ----------
        method : str
            Aggregation method
        specific_months : list of int, optional
            Specific months to include (1-12)
        years : list of int, optional
            Specific years to include

        Returns:
        -------
        xr.Dataset
            Monthly aggregated dataset
        """
        self._validate_method(method)

        data_with_coords = self._add_time_coordinates(self.dataset)

        # Filter by specific months if provided
        if specific_months:
            data_with_coords = data_with_coords.where(
                data_with_coords.month.isin(specific_months), drop=True
            )

        # Filter by specific years if provided
        if years:
            data_with_coords = data_with_coords.where(
                data_with_coords.year.isin(years), drop=True
            )

        if method == "count":
            result = data_with_coords.groupby("time.month").count()
        else:
            agg_func = getattr(data_with_coords.groupby("time.month"), method)
            result = agg_func()

        return result

    def annual_aggregation(
        self, method: str = "mean", specific_years: list[int] | None = None
    ) -> xr.Dataset:
        """Aggregate to annual values.

        Parameters
        ----------
        method : str
            Aggregation method
        specific_years : list of int, optional
            Specific years to include

        Returns:
        -------
        xr.Dataset
            Annual aggregated dataset
        """
        self._validate_method(method)

        data_with_coords = self._add_time_coordinates(self.dataset)

        # Filter by specific years if provided
        if specific_years:
            data_with_coords = data_with_coords.where(
                data_with_coords.year.isin(specific_years), drop=True
            )

        if method == "count":
            result = data_with_coords.groupby("time.year").count()
        else:
            agg_func = getattr(data_with_coords.groupby("time.year"), method)
            result = agg_func()

        return result

    def seasonal_aggregation(
        self,
        method: str = "mean",
        seasons: list[str] | None = None,
        years: list[int] | None = None,
    ) -> xr.Dataset:
        """Aggregate to seasonal values.

        Parameters
        ----------
        method : str
            Aggregation method
        seasons : list of str, optional
            Specific seasons to include ('winter', 'spring', 'summer', 'autumn')
        years : list of int, optional
            Specific years to include

        Returns:
        -------
        xr.Dataset
            Seasonal aggregated dataset
        """
        self._validate_method(method)

        data_with_coords = self._add_time_coordinates(self.dataset)

        # Filter by specific seasons if provided
        if seasons:
            valid_seasons = [s for s in seasons if s in self.SEASONS.keys()]
            if not valid_seasons:
                raise ValueError(
                    f"No valid seasons provided. Available: {list(self.SEASONS.keys())}"
                )

            data_with_coords = data_with_coords.where(
                data_with_coords.season.isin(valid_seasons), drop=True
            )

        # Filter by specific years if provided
        if years:
            data_with_coords = data_with_coords.where(
                data_with_coords.year.isin(years), drop=True
            )

        if method == "count":
            result = data_with_coords.groupby("season").count()
        else:
            agg_func = getattr(data_with_coords.groupby("season"), method)
            result = agg_func()

        return result

    def custom_time_aggregation(
        self,
        time_periods: list[tuple[str, str]],
        method: str = "mean",
        period_names: list[str] | None = None,
    ) -> xr.Dataset:
        """Aggregate over custom time periods.

        Parameters
        ----------
        time_periods : list of tuples
            List of (start_date, end_date) tuples in 'YYYY-MM-DD' format
        method : str
            Aggregation method
        period_names : list of str, optional
            Names for each period

        Returns:
        -------
        xr.Dataset
            Custom period aggregated dataset
        """
        self._validate_method(method)

        if period_names and len(period_names) != len(time_periods):
            raise ValueError("Number of period names must match number of time periods")

        results = []

        for i, (start_date, end_date) in enumerate(time_periods):
            period_data = self.dataset.sel(time=slice(start_date, end_date))

            if method == "count":
                agg_data = period_data.count(dim="time")
            else:
                agg_func = getattr(period_data, method)
                agg_data = agg_func(dim="time")

            # Add period identifier
            period_name = period_names[i] if period_names else f"period_{i + 1}"
            agg_data.coords["period"] = period_name

            results.append(agg_data)

        # Combine results
        combined = xr.concat(results, dim="period")
        return combined

    def rolling_aggregation(
        self,
        window: int,
        method: str = "mean",
        center: bool = True,
        min_periods: int | None = None,
    ) -> xr.Dataset:
        """Apply rolling window aggregation.

        Parameters
        ----------
        window : int
            Size of the rolling window (in time steps)
        method : str
            Aggregation method
        center : bool
            Whether to center the window
        min_periods : int, optional
            Minimum number of observations required

        Returns:
        -------
        xr.Dataset
            Rolling aggregated dataset
        """
        self._validate_method(method)

        rolling_obj = self.dataset.rolling(
            time=window, center=center, min_periods=min_periods
        )

        if method == "count":
            result = rolling_obj.count()
        else:
            agg_func = getattr(rolling_obj, method)
            result = agg_func()

        return result

    def time_groupby_aggregation(
        self, groupby_freq: str, method: str = "mean"
    ) -> xr.Dataset:
        """Group by time frequency and aggregate.

        Parameters
        ----------
        groupby_freq : str
            Pandas-style frequency string (e.g., 'M' for month, 'Q' for quarter, 'Y' for year)
        method : str
            Aggregation method

        Returns:
        -------
        xr.Dataset
            Frequency-grouped aggregated dataset
        """
        self._validate_method(method)

        if method == "count":
            result = self.dataset.resample(time=groupby_freq).count()
        else:
            agg_func = getattr(self.dataset.resample(time=groupby_freq), method)
            result = agg_func()

        return result

    def get_time_before_date(self, reference_date: str, days_before: int) -> xr.Dataset:
        """Get data for a specific number of days before a reference date.

        Parameters
        ----------
        reference_date : str
            Reference date in 'YYYY-MM-DD' format
        days_before : int
            Number of days before the reference date

        Returns:
        -------
        xr.Dataset
            Dataset for the specified period
        """
        ref_date = pd.to_datetime(reference_date)
        start_date = ref_date - pd.Timedelta(days=days_before)

        return self.dataset.sel(time=slice(start_date, ref_date))

    def aggregate_multiple_periods(
        self, periods_config: list[dict]
    ) -> dict[str, xr.Dataset]:
        """Aggregate multiple periods with different configurations.

        Parameters
        ----------
        periods_config : list of dict
            List of configuration dictionaries, each containing:
            - 'type': aggregation type ('monthly', 'annual', 'seasonal', 'custom')
            - 'method': aggregation method
            - Additional parameters specific to each type

        Returns:
        -------
        dict
            Dictionary of aggregated datasets keyed by period names
        """
        results = {}

        for i, config in enumerate(periods_config):
            agg_type = config.get("type", "monthly")
            method = config.get("method", "mean")
            name = config.get("name", f"{agg_type}_{i + 1}")

            if agg_type == "monthly":
                result = self.monthly_aggregation(
                    method=method,
                    specific_months=config.get("months"),
                    years=config.get("years"),
                )
            elif agg_type == "annual":
                result = self.annual_aggregation(
                    method=method, specific_years=config.get("years")
                )
            elif agg_type == "seasonal":
                result = self.seasonal_aggregation(
                    method=method,
                    seasons=config.get("seasons"),
                    years=config.get("years"),
                )
            elif agg_type == "custom":
                result = self.custom_time_aggregation(
                    time_periods=config.get("time_periods", []),
                    method=method,
                    period_names=config.get("period_names"),
                )
            else:
                logger.warning(f"Unknown aggregation type: {agg_type}")
                continue

            results[name] = result

        return results
