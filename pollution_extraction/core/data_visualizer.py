"""Data visualization module for pollution data."""

import logging
import warnings
from pathlib import Path
from typing import Any

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr

logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)


class DataVisualizer:
    """Class for visualizing pollution data.

    Provides various plotting methods including maps, time series,
    distributions, and comparative analyses.
    """

    # Color palettes for different pollution types
    POLLUTION_COLORMAPS = {
        "bc": "plasma",
        "no2": "viridis",
        "pm25": "Reds",
        "pm10": "Oranges",
        "default": "viridis",
    }

    def __init__(
        self,
        dataset: xr.Dataset,
        pollution_variable: str,
        pollution_type: str = "default",
    ):
        """Initialize the data visualizer.

        Parameters
        ----------
        dataset : xr.Dataset
            The pollution dataset
        pollution_variable : str
            Name of the pollution variable to visualize
        pollution_type : str
            Type of pollution for color mapping
        """
        self.dataset = dataset
        self.pollution_variable = pollution_variable
        self.pollution_type = pollution_type
        self.cmap = self.POLLUTION_COLORMAPS.get(pollution_type, "viridis")

        # Set style
        plt.style.use("seaborn-v0_8")
        sns.set_palette("husl")

    def _maybe_invert_yaxis(self, ax, data):
        """Invert y-axis if y-coordinates are descending (to match imshow origin='upper')."""
        y = data.y.values if hasattr(data, "y") else None
        if y is not None and len(y) > 1 and y[0] > y[-1]:
            ax.invert_yaxis()

    def plot_spatial_map(
        self,
        time_index: int | str = 0,
        figsize: tuple[int, int] = (12, 8),
        title: str | None = None,
        save_path: str | Path | None = None,
        show_colorbar: bool = True,
        vmin: float | None = None,
        vmax: float | None = None,
        projection: Any | None = None,
    ):
        """Create a spatial map of pollution data.

        Parameters
        ----------
        time_index : int or str
            Time index or date string to plot
        figsize : tuple
            Figure size (width, height)
        title : str, optional
            Plot title
        save_path : str or Path, optional
            Path to save the figure
        show_colorbar : bool
            Whether to show colorbar
        vmin, vmax : float, optional
            Color scale limits
        projection : cartopy projection, optional
            Map projection

        Returns:
        -------
        plt.Figure
            The created figure
        """
        # Select data for the specified time
        if isinstance(time_index, str):
            data = self.dataset[self.pollution_variable].sel(
                time=time_index, method="nearest"
            )
        else:
            data = self.dataset[self.pollution_variable].isel(time=time_index)

        # Create figure
        if projection:
            fig = plt.figure(figsize=figsize)
            ax = plt.axes(projection=projection)
            ax.add_feature(cfeature.COASTLINE)
            ax.add_feature(cfeature.BORDERS)
            ax.add_feature(cfeature.LAND, alpha=0.3)
            ax.add_feature(cfeature.OCEAN, alpha=0.3)
        else:
            fig, ax = plt.subplots(figsize=figsize)

        # Plot data
        data.plot(
            ax=ax,
            cmap=self.cmap,
            add_colorbar=show_colorbar,
            vmin=vmin,
            vmax=vmax,
            transform=ccrs.PlateCarree() if projection else None,
        )
        # Invert y-axis if needed for correct orientation
        if not projection:
            self._maybe_invert_yaxis(ax, data)

        # Set title
        if title is None:
            time_str = pd.to_datetime(data.time.values).strftime("%Y-%m-%d")
            title = f"{self.pollution_variable.replace('_', ' ').title()} - {time_str}"

        ax.set_title(title, fontsize=14, fontweight="bold")

        if not projection:
            ax.set_xlabel("X Coordinate (m)", fontsize=12)
            ax.set_ylabel("Y Coordinate (m)", fontsize=12)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Map saved to {save_path}")

        return fig

    def plot_time_series(
        self,
        location: dict[str, float] | None = None,
        spatial_aggregation: str = "mean",
        figsize: tuple[int, int] = (12, 6),
        title: str | None = None,
        save_path: str | Path | None = None,
    ) -> plt.Figure:
        """Create a time series plot.

        Parameters
        ----------
        location : dict, optional
            Location dictionary with 'x' and 'y' keys for point extraction
        spatial_aggregation : str
            Spatial aggregation method if no location specified
        figsize : tuple
            Figure size
        title : str, optional
            Plot title
        save_path : str or Path, optional
            Path to save the figure

        Returns:
        -------
        plt.Figure
            The created figure
        """
        # Get time series data
        if location:
            data = self.dataset[self.pollution_variable].sel(
                x=location["x"], y=location["y"], method="nearest"
            )
            location_str = f"at ({location['x']:.0f}, {location['y']:.0f})"
        else:
            if spatial_aggregation == "mean":
                data = self.dataset[self.pollution_variable].mean(dim=["x", "y"])
            elif spatial_aggregation == "max":
                data = self.dataset[self.pollution_variable].max(dim=["x", "y"])
            elif spatial_aggregation == "min":
                data = self.dataset[self.pollution_variable].min(dim=["x", "y"])
            location_str = f"({spatial_aggregation} over domain)"

        # Create plot
        fig, ax = plt.subplots(figsize=figsize)

        data.plot(ax=ax, linewidth=2, color="steelblue")

        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45)

        # Set labels and title
        if title is None:
            title = f"{self.pollution_variable.replace('_', ' ').title()} Time Series {location_str}"

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Time", fontsize=12)
        ax.set_ylabel(
            f"{self.pollution_variable.replace('_', ' ').title()}", fontsize=12
        )
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Time series plot saved to {save_path}")

        return fig

    def plot_seasonal_cycle(
        self,
        spatial_aggregation: str = "mean",
        figsize: tuple[int, int] = (10, 6),
        title: str | None = None,
        save_path: str | Path | None = None,
    ) -> plt.Figure:
        """Create a seasonal cycle plot.

        Parameters
        ----------
        spatial_aggregation : str
            Spatial aggregation method
        figsize : tuple
            Figure size
        title : str, optional
            Plot title
        save_path : str or Path, optional
            Path to save the figure

        Returns:
        -------
        plt.Figure
            The created figure
        """
        # Aggregate spatially
        if spatial_aggregation == "mean":
            data = self.dataset[self.pollution_variable].mean(dim=["x", "y"])
        elif spatial_aggregation == "max":
            data = self.dataset[self.pollution_variable].max(dim=["x", "y"])
        elif spatial_aggregation == "min":
            data = self.dataset[self.pollution_variable].min(dim=["x", "y"])

        # Group by month
        monthly_data = data.groupby("time.month").mean()
        monthly_std = data.groupby("time.month").std()

        # Create plot
        fig, ax = plt.subplots(figsize=figsize)

        months = monthly_data.month.values
        values = monthly_data.values
        std_values = monthly_std.values

        ax.plot(
            months, values, marker="o", linewidth=2, markersize=8, color="steelblue"
        )
        ax.fill_between(
            months,
            values - std_values,
            values + std_values,
            alpha=0.3,
            color="steelblue",
        )

        # Customize plot
        ax.set_xlabel("Month", fontsize=12)
        ax.set_ylabel(
            f"{self.pollution_variable.replace('_', ' ').title()}", fontsize=12
        )
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(
            [
                "Jan",
                "Feb",
                "Mar",
                "Apr",
                "May",
                "Jun",
                "Jul",
                "Aug",
                "Sep",
                "Oct",
                "Nov",
                "Dec",
            ]
        )
        ax.grid(True, alpha=0.3)

        if title is None:
            title = (
                f"Seasonal Cycle - {self.pollution_variable.replace('_', ' ').title()}"
            )
        ax.set_title(title, fontsize=14, fontweight="bold")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Seasonal cycle plot saved to {save_path}")

        return fig

    def plot_distribution(
        self,
        time_subset: slice | None = None,
        spatial_subset: dict | None = None,
        bins: int = 50,
        figsize: tuple[int, int] = (10, 6),
        title: str | None = None,
        save_path: str | Path | None = None,
    ) -> plt.Figure:
        """Create a distribution plot (histogram).

        Parameters
        ----------
        time_subset : slice, optional
            Time subset for analysis
        spatial_subset : dict, optional
            Spatial subset bounds
        bins : int
            Number of histogram bins
        figsize : tuple
            Figure size
        title : str, optional
            Plot title
        save_path : str or Path, optional
            Path to save the figure

        Returns:
        -------
        plt.Figure
            The created figure
        """
        data = self.dataset[self.pollution_variable]

        # Apply subsets
        if time_subset:
            data = data.sel(time=time_subset)

        if spatial_subset:
            data = data.sel(
                x=slice(spatial_subset.get("minx"), spatial_subset.get("maxx")),
                y=slice(spatial_subset.get("miny"), spatial_subset.get("maxy")),
            )

        # Flatten data and remove NaNs
        values = data.values.flatten()
        values = values[~np.isnan(values)]

        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Histogram
        ax1.hist(values, bins=bins, alpha=0.7, edgecolor="black", color="steelblue")
        ax1.set_xlabel(
            f"{self.pollution_variable.replace('_', ' ').title()}", fontsize=12
        )
        ax1.set_ylabel("Frequency", fontsize=12)
        ax1.grid(True, alpha=0.3)

        # Box plot
        ax2.boxplot(
            values,
            vert=True,
            patch_artist=True,
            boxprops=dict(facecolor="lightblue", alpha=0.7),
        )
        ax2.set_ylabel(
            f"{self.pollution_variable.replace('_', ' ').title()}", fontsize=12
        )
        ax2.grid(True, alpha=0.3)

        if title is None:
            title = (
                f"Distribution - {self.pollution_variable.replace('_', ' ').title()}"
            )
        fig.suptitle(title, fontsize=14, fontweight="bold")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Distribution plot saved to {save_path}")

        return fig

    def plot_spatial_statistics(
        self,
        statistic: str = "mean",
        figsize: tuple[int, int] = (12, 8),
        title: str | None = None,
        save_path: str | Path | None = None,
    ):
        """Create a spatial map of temporal statistics.

        Parameters
        ----------
        statistic : str
            Temporal statistic to compute ('mean', 'max', 'min', 'std')
        figsize : tuple
            Figure size
        title : str, optional
            Plot title
        save_path : str or Path, optional
            Path to save the figure

        Returns:
        -------
        plt.Figure
            The created figure
        """
        # Compute temporal statistic
        data = self.dataset[self.pollution_variable]

        if statistic == "mean":
            stat_data = data.mean(dim="time")
        elif statistic == "max":
            stat_data = data.max(dim="time")
        elif statistic == "min":
            stat_data = data.min(dim="time")
        elif statistic == "std":
            stat_data = data.std(dim="time")
        else:
            raise ValueError(f"Unsupported statistic: {statistic}")

        # Create plot
        fig, ax = plt.subplots(figsize=figsize)

        stat_data.plot(ax=ax, cmap=self.cmap, add_colorbar=True)
        # Invert y-axis if needed for correct orientation
        self._maybe_invert_yaxis(ax, stat_data)

        if title is None:
            title = f"Temporal {statistic.title()} - {self.pollution_variable.replace('_', ' ').title()}"

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("X Coordinate (m)", fontsize=12)
        ax.set_ylabel("Y Coordinate (m)", fontsize=12)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Spatial statistics plot saved to {save_path}")

        return fig

    def plot_comparison(
        self,
        other_dataset: xr.Dataset,
        other_variable: str,
        comparison_type: str = "difference",
        time_index: int | str = 0,
        figsize: tuple[int, int] = (15, 5),
        title: str | None = None,
        save_path: str | Path | None = None,
    ) -> plt.Figure:
        """Create a comparison plot between two datasets.

        Parameters
        ----------
        other_dataset : xr.Dataset
            Second dataset for comparison
        other_variable : str
            Variable name in the second dataset
        comparison_type : str
            Type of comparison ('difference', 'ratio', 'side_by_side')
        time_index : int or str
            Time index or date string to plot
        figsize : tuple
            Figure size
        title : str, optional
            Plot title
        save_path : str or Path, optional
            Path to save the figure

        Returns:
        -------
        plt.Figure
            The created figure
        """
        # Select data for the specified time
        if isinstance(time_index, str):
            data1 = self.dataset[self.pollution_variable].sel(
                time=time_index, method="nearest"
            )
            data2 = other_dataset[other_variable].sel(time=time_index, method="nearest")
        else:
            data1 = self.dataset[self.pollution_variable].isel(time=time_index)
            data2 = other_dataset[other_variable].isel(time=time_index)

        if comparison_type == "side_by_side":
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

            # Plot first dataset
            data1.plot(ax=ax1, cmap=self.cmap, add_colorbar=True)
            ax1.set_title("Dataset 1", fontsize=12)

            # Plot second dataset
            data2.plot(ax=ax2, cmap=self.cmap, add_colorbar=True)
            ax2.set_title("Dataset 2", fontsize=12)

        else:
            fig, ax = plt.subplots(figsize=(10, 8))

            if comparison_type == "difference":
                diff_data = data1 - data2
                diff_data.plot(ax=ax, cmap="RdBu_r", add_colorbar=True, center=0)
                comparison_title = "Difference (Dataset1 - Dataset2)"

            elif comparison_type == "ratio":
                ratio_data = data1 / data2
                ratio_data.plot(ax=ax, cmap="RdYlBu_r", add_colorbar=True)
                comparison_title = "Ratio (Dataset1 / Dataset2)"

            ax.set_title(comparison_title, fontsize=12)

        if title is None:
            time_str = pd.to_datetime(data1.time.values).strftime("%Y-%m-%d")
            title = f"Dataset Comparison - {time_str}"

        fig.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Comparison plot saved to {save_path}")

        return fig

    def create_animation(
        self,
        output_path: str | Path,
        time_step: int = 1,
        fps: int = 5,
        dpi: int = 100,
    ) -> None:
        """Create an animation of the spatial data over time.

        Parameters
        ----------
        output_path : str or Path
            Path for the output animation file
        time_step : int
            Time step interval for animation frames
        fps : int
            Frames per second
        dpi : int
            Resolution of the animation
        """
        try:
            from matplotlib.animation import FuncAnimation

            data = self.dataset[self.pollution_variable]

            # Set up the figure and axis
            fig, ax = plt.subplots(figsize=(10, 8))

            # Get data range for consistent color scale
            vmin = float(data.min())
            vmax = float(data.max())

            def animate(frame):
                ax.clear()
                time_data = data.isel(time=frame * time_step)

                im = time_data.plot(
                    ax=ax, cmap=self.cmap, vmin=vmin, vmax=vmax, add_colorbar=False
                )

                time_str = pd.to_datetime(time_data.time.values).strftime("%Y-%m-%d")
                ax.set_title(
                    f"{self.pollution_variable.replace('_', ' ').title()} - {time_str}",
                    fontsize=14,
                    fontweight="bold",
                )

                return [im]

            # Create animation
            frames = len(data.time) // time_step
            anim = FuncAnimation(fig, animate, frames=frames, blit=False, repeat=True)

            # Save animation
            anim.save(output_path, fps=fps, dpi=dpi, writer="pillow")
            logger.info(f"Animation saved to {output_path}")

            plt.close(fig)

        except ImportError:
            logger.error("Animation requires matplotlib with pillow or ffmpeg")
            raise
