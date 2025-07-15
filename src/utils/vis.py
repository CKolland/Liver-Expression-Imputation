from dataclasses import dataclass
import re
import warnings

import numpy as np
import pandas as pd
from plotnine import *
from plotnine.exceptions import PlotnineWarning
import scipy.sparse as sparse

import utils._constants as C

# Suppress Plotnine warnings
warnings.filterwarnings("ignore", category=PlotnineWarning)


@dataclass
class PlotConfig:
    """Configuration class for plot styling and behavior.

    This dataclass contains all the configuration options for customizing
    plots created by the PeriodPlotter class. It provides sensible defaults
    for all parameters while allowing fine-grained control over plot appearance,
    styling, and behavior.

    The configuration is organized into logical groups:
    - Basic plot settings (plot type, smoothing)
    - Visual styling (colors, sizes, transparency)
    - Layout and appearance (figure size, background, grid)
    - Text styling (font sizes, angles)
    - Labels (titles, axis labels)
    - Advanced features (faceting, axis limits)
    """

    # Basic plot settings
    plot_type: str = "line"  # Type of plot geometry to use
    smooth: bool = False  # Whether to add a smoothing layer
    confidence_interval: bool = False  # Whether to add confidence for smoothing

    # Visual styling
    point_size: float = 1.5  # Size of points in the plot
    line_size: float = 0.8  # Thickness of lines in the plot
    alpha: float = 0.8  # Transparency level for plot elements

    # - A string for a ColorBrewer palette name (e.g., "Paired", "Set1")
    # - A list of hex color codes (e.g., ["#FF0000", "#00FF00"])
    # - A dictionary mapping variable names to hex colors
    colors: str | list[str] | dict[str, str] = "Paired"

    # Layout and appearance
    figure_size: tuple[int, int] = (12, 8)  # Dimensions in inches (width, height)
    background_color: str = "white"
    grid: bool = True  # Whether to display grid lines on the plot.
    legend_position: str = "right"

    # Text styling
    text_size: int = 11
    title_size: int = 14
    axis_text_angle: float = 0  # Rotation for x-axis text labels in degrees

    # Labels
    title: str = "Traning vs. Validation Loss per Fold"
    x_label: str = "Epochs"
    y_label: str = "Loss"
    legend_title: str = "Loss"

    # Advanced features
    facet_by: str | None = None  # Column name to use for faceting (creating subplots)
    y_limits: tuple[float, float] | None = None


class PeriodPlotter:
    """A flexible plotting class for period-based data visualization.

    This class creates customizable plots showing how values change over periods,
    with support for multiple series, faceting, smoothing, and extensive styling options.
    It uses plotnine (ggplot2 for Python) as the underlying plotting library.

    :param data: The input dataset containing the data to plot
    :type data: pd.DataFrame
    :param period_col: Column name for the x-axis (time periods, epochs, etc.)
    :type period_col: str
    :param value_cols: Column name(s) for the y-axis values
    :type value_cols: str | list[str]

    :raises ValueError: If data is not a DataFrame, or if specified columns are not found

    ## Example

    .. code-block:: python
        plotter = PeriodPlotter(data, 'period', ['metric1', 'metric2'])
        plot = plotter.plot()
    """

    def __init__(
        self,
        data: pd.DataFrame,
        period_col: str,
        value_cols: str | list[str],
    ):
        """Initialize the plotter with data and column specifications.

        :param data: The input dataset containing the data to plot
        :type data: pd.DataFrame
        :param period_col: Column name for the x-axis (time periods, epochs, etc.)
        :type period_col: str
        :param value_cols: Column name(s) for the y-axis values
        :type value_cols: str | list[str]

        :raises ValueError: If data is not a DataFrame, or if specified columns are not found
        """
        self.data = data
        self.period_col = period_col
        self.value_cols = self._transform_value_cols(value_cols)
        self.is_multi: bool = len(self.value_cols) > 1

        # Initialize plot data and plot (will be set during plotting)
        self.to_plot = None
        self.plot = None

        # Validate inputs during initialization
        self._validate_inputs()

    @staticmethod
    def _transform_value_cols(value_cols: str | list[str]) -> list[str]:
        """Convert value_columns to a list format for uniform processing.

        :param value_cols: Column name(s) for the y-axis values
        :type value_cols: str | list[str]

        :return: List of value column names
        :rtype: list[str]
        """
        if isinstance(value_cols, str):
            return [value_cols]
        return value_cols

    def _validate_inputs(self):
        """Validate inputs and raise appropriate errors.

        Checks that:
        - Data is a pandas DataFrame
        - Period column exists in the data
        - All value columns exist in the data

        :raises ValueError: If validation fails
        """
        if not isinstance(self.data, pd.DataFrame):
            raise ValueError(C.DATA_NOT_DF_ERR)

        # Validate that period column exists
        if self.period_col not in self.data.columns:
            raise ValueError(C.PERIOD_COL_NOT_FOUND_ERR.format(col=self.period_col))

        # Validate that all value columns exist
        missing_cols = [col for col in self.value_cols if col not in self.data.columns]
        if missing_cols:
            raise ValueError(C.VALUE_COLS_NOT_FOUND_ERR.format(missing=missing_cols))

    def _prep_plot_data(self, facet_by: str | None):
        """Prepare data for plotting, reshaping if necessary for multiple series.

        For multiple series, converts data from wide to long format to create a single
        value column with a 'variable' column indicating the series.

        :param facet_by: Column name to use for faceting, or None
        :type facet_by: str | None

        :raises ValueError: If facet_by column is not found in the data
        """
        # Convert data from wide to long format for multiple series
        if self.is_multi:
            # Columns to keep when reshaping
            id_vars = [self.period_col]
            if facet_by is not None:
                if facet_by in self.data.columns:
                    id_vars.append(facet_by)
                else:
                    raise ValueError(C.FACET_COL_NOT_FOUND_ERR.format(col=facet_by))

            self.to_plot = self.data.melt(
                id_vars=id_vars,
                value_vars=self.value_cols,
                var_name="variable",
                value_name="value",
            )
        else:
            self.to_plot = self.data.copy()

    def _create_base_plot(self, config: PlotConfig):
        """Create the base ggplot object with appropriate aesthetics.

        :param config: Configuration object containing plot settings
        :type config: PlotConfig
        """
        if self.is_multi:
            # Multiple series: map color to variable for differentiation
            base_plot = ggplot(
                self.to_plot,
                aes(x=self.period_col, y="value", color="variable"),
            )
        else:
            # Single series: no color mapping in base aesthetics
            base_plot = ggplot(
                self.to_plot,
                aes(x=self.period_col, y=self.value_cols[0]),
            )

        # Add labels
        base_plot = base_plot + labs(
            title=config.title,
            x=config.x_label,
            y=config.y_label,
        )

        self.plot = base_plot

    def _add_geoms(self, config: PlotConfig):
        """Add geometric layers (lines and/or points) to the plot.

        :param config: Configuration object containing plot settings
        :type config: PlotConfig

        :raises ValueError: If plot_type is not in ALLOWED_GEOMS
        """
        # Validate plot_type
        if config.plot_type not in C.ALLOWED_GEOMS:
            raise ValueError(C.UNKNOWN_GEOM_ERR)

        # Add line geometry if requested
        if config.plot_type in ["line", "both"]:
            self.plot = self.plot + geom_line(size=config.line_size, alpha=config.alpha)

        # Add point geometry if requested
        if config.plot_type in ["point", "both"]:
            self.plot = self.plot + geom_point(
                size=config.point_size,
                alpha=config.alpha,
            )

    @staticmethod
    def _validate_colors(colors: str | list[str]) -> bool:
        """Validate that all given colors are in valid hexadecimal format.

        :param colors: A single color string or a list of color strings.
        :type colors: str | list[str]

        :return: True if all colors are valid hex codes, False otherwise.
        :rtype: bool
        """
        # Regular expression pattern for hex color validation:
        # 1. ^: Start of string
        # 2. #: Hash symbol
        # 3. [A-Fa-f0-9]{n}: Hexadecimal characters of length n
        # 4. $: End of string
        hex_pattern = r"^#([A-Fa-f0-9]{3}|[A-Fa-f0-9]{6}|[A-Fa-f0-9]{8})$"

        if isinstance(colors, str):
            colors = [colors]  # Convert to list for uniform processing

        return all(re.match(hex_pattern, color) for color in colors)

    def _apply_colors(self, config: PlotConfig):
        """Apply color specifications to the plot.

        Handles both custom hex colors and color brewer palettes.

        :param config: Configuration object containing color settings
        :type config: PlotConfig
        """

        if self._validate_colors(config.colors):
            # Handle custom hex colors
            if isinstance(config.colors, dict):
                color_map = config.colors
            elif isinstance(config.colors, list):
                # Map colors to value columns
                color_map = dict(
                    zip(self.value_cols, config.colors[: len(self.value_cols)]),
                )
            else:
                color_map = [config.colors]

            self.plot = self.plot + scale_color_manual(
                values=color_map, name=config.legend_title
            )
        else:
            # Use color brewer palette
            self.plot = self.plot + scale_color_brewer(
                type="qual",
                palette=config.colors,
                name=config.legend_title,
            )

    def _add_smoothing(self, config: PlotConfig):
        """Add smoothing layer to the plot.

        :param config: Configuration object containing smoothing settings
        :type config: PlotConfig
        """
        if config.smooth:
            smooth_args = {
                "se": config.confidence_interval,
                "alpha": config.alpha * 0.3,
            }
            self.plot = self.plot + geom_smooth(**smooth_args)

    def _apply_theme_and_layout(self, config: PlotConfig):
        """Apply theme and layout settings to the plot.

        :param config: Configuration object containing theme settings
        :type config: PlotConfig
        """

        # Build theme elements dictionary
        theme_elements = {
            "figure_size": config.figure_size,
            "plot_title": element_text(size=config.title_size, weight="bold"),
            "axis_text": element_text(size=config.text_size),
            "axis_title": element_text(size=config.text_size + 1),
            "legend_title": element_text(size=config.text_size),
            "legend_text": element_text(size=config.text_size - 1),
            "panel_background": element_rect(fill=config.background_color),
            "plot_background": element_rect(fill=config.background_color),
            "legend_position": config.legend_position,
        }

        # Apply conditional theme elements
        if not config.grid:
            theme_elements["panel_grid"] = element_blank()

        if config.axis_text_angle != 0:
            theme_elements["axis_text_x"] = element_text(
                angle=config.axis_text_angle, hjust=1
            )

        self.plot = self.plot + theme_minimal()
        self.plot = self.plot + theme(**theme_elements)

    def _apply_faceting(self, config: PlotConfig):
        """Add faceting if requested.

        :param config: Configuration object containing faceting settings
        :type config: PlotConfig
        """
        if config.facet_by and config.facet_by in self.data.columns:
            self.plot = self.plot + facet_wrap(f"~{config.facet_by}")

    def _apply_axis_limits(self, config: PlotConfig):
        """Set axis limits if provided.

        :param config: Configuration object containing axis limit settings
        :type config: PlotConfig
        """
        if config.y_limits:
            self.plot = self.plot + scale_y_continuous(limits=config.y_limits)

    def render_plot(self, config: PlotConfig | None = None, **kwargs) -> ggplot:
        """Create the plot with the specified configuration.

        This is the main plotting method that orchestrates all plot creation steps.
        It prepares the data, creates the base plot, adds geometric layers, applies
        styling, and returns the final plot object.

        :param config: Configuration object with plot settings. If None, uses default config
        :type config: PlotConfig | None
        :param kwargs: Individual configuration parameters that override config settings
        :type kwargs: Any

        :return: The constructed plotnine plot object
        :rtype: ggplot

        :raises ValueError: If unknown configuration parameter is provided in kwargs

        ## Example

        .. code-block:: python
            plotter = PeriodPlotter(data, 'period', 'value')
            plot = plotter.plot(title='My Plot', plot_type='line')
        """
        # Create config with defaults if not provided
        if config is None:
            config = PlotConfig()

        # Apply any keyword argument overrides
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                raise ValueError(C.UNKNOWN_CONFIG_ARG_ERR.foramt(key=key))

        # Prepare data for plotting
        self._prep_plot_data(config.facet_by)

        # Build plot step by step
        self._create_base_plot(config)
        self._add_geoms(config)
        self._apply_colors(config)
        self._add_smoothing(config)
        self._apply_theme_and_layout(config)
        self._apply_faceting(config)
        self._apply_axis_limits(config)

        return self.plot


class Plotter:
    """
    Create a class plotter that gets a PlotConfig and has methods that just get the data to plot.
    Can be combined with class on top.
    """

    pass


def _calc_sparse_mean_expression(matrix):
    """_summary_"""
    # Convert to CSC format for efficient column operations
    if not sparse.isspmatrix_csc(matrix):
        matrix = matrix.tocsc()

    # Calculate mean for each column
    mean_expr = np.array(matrix.mean(axis=0)).flatten()

    return mean_expr


def plot_target_vs_prediction(
    targets,
    predictions,
    save_file,
    labels: tuple[str, str] = ("Targets", "Predictions"),
):
    """_summary_"""

    mean_targets = _calc_sparse_mean_expression(targets)
    mean_predictions = _calc_sparse_mean_expression(predictions)

    # Create DataFrame
    df = pd.DataFrame({"targets": mean_targets, "predictions": mean_predictions})

    # Start building the plot
    p = ggplot(df, aes(x="targets", y="predictions")) + geom_point()

    # Add diagonal
    min_val = min(mean_targets.min(), mean_predictions.min())
    max_val = max(mean_targets.max(), mean_predictions.max())
    diagonal_df = pd.DataFrame({"x": [min_val, max_val], "y": [min_val, max_val]})
    p = p + geom_line(
        data=diagonal_df,
        mapping=aes(x="x", y="y"),
        color="blue",
        linetype="dashed",
        alpha=0.7,
        size=0.8,
    )

    # Add labels and title
    p = p + labs(
        x=labels[0], y=labels[1], title="Mean Expression for Targets vs. Predictions"
    )
    p = (
        p
        + theme_minimal()
        + theme(
            panel_background=element_rect(fill="white"),
            plot_background=element_rect(fill="white"),
        )
    )

    ggsave(p, save_file, dpi=300)
