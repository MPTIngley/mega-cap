"""Chart creation utilities with clean, publication-quality styling."""

from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class ChartTheme:
    """
    Clean, publication-quality chart theme.

    White background, large fonts, clear axes.
    NO default matplotlib garbage.
    """

    # Colors
    BACKGROUND = "#FFFFFF"
    GRID_COLOR = "#E5E5E5"
    TEXT_COLOR = "#2C3E50"
    POSITIVE_COLOR = "#27AE60"
    NEGATIVE_COLOR = "#E74C3C"
    PRIMARY_COLOR = "#3498DB"
    SECONDARY_COLOR = "#9B59B6"
    NEUTRAL_COLOR = "#95A5A6"

    # Fonts
    FONT_FAMILY = "Arial, sans-serif"
    TITLE_SIZE = 18
    AXIS_TITLE_SIZE = 14
    TICK_SIZE = 12
    LEGEND_SIZE = 12

    @classmethod
    def get_layout(cls, title: str = "", height: int = 500) -> dict:
        """Get standard layout configuration."""
        return {
            "title": {
                "text": title,
                "font": {"size": cls.TITLE_SIZE, "color": cls.TEXT_COLOR, "family": cls.FONT_FAMILY},
                "x": 0.5,
                "xanchor": "center"
            },
            "paper_bgcolor": cls.BACKGROUND,
            "plot_bgcolor": cls.BACKGROUND,
            "font": {"family": cls.FONT_FAMILY, "color": cls.TEXT_COLOR},
            "height": height,
            "margin": {"l": 60, "r": 40, "t": 60, "b": 60},
            "xaxis": {
                "gridcolor": cls.GRID_COLOR,
                "linecolor": cls.TEXT_COLOR,
                "tickfont": {"size": cls.TICK_SIZE},
                "title_font": {"size": cls.AXIS_TITLE_SIZE}
            },
            "yaxis": {
                "gridcolor": cls.GRID_COLOR,
                "linecolor": cls.TEXT_COLOR,
                "tickfont": {"size": cls.TICK_SIZE},
                "title_font": {"size": cls.AXIS_TITLE_SIZE}
            },
            "legend": {
                "font": {"size": cls.LEGEND_SIZE},
                "bgcolor": "rgba(255,255,255,0.8)",
                "bordercolor": cls.GRID_COLOR,
                "borderwidth": 1
            },
            "hovermode": "x unified"
        }


def create_price_chart(
    df: pd.DataFrame,
    ticker: str,
    show_volume: bool = True,
    signals: list[dict] | None = None
) -> go.Figure:
    """
    Create a candlestick price chart with optional volume and signals.

    Args:
        df: DataFrame with OHLCV data
        ticker: Stock ticker for title
        show_volume: Whether to show volume subplot
        signals: List of signal dictionaries to mark on chart

    Returns:
        Plotly Figure
    """
    if show_volume:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3]
        )
    else:
        fig = go.Figure()

    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df["date"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Price",
            increasing_line_color=ChartTheme.POSITIVE_COLOR,
            decreasing_line_color=ChartTheme.NEGATIVE_COLOR
        ),
        row=1 if show_volume else None,
        col=1 if show_volume else None
    )

    # Volume bars
    if show_volume and "volume" in df.columns:
        colors = [
            ChartTheme.POSITIVE_COLOR if row["close"] >= row["open"]
            else ChartTheme.NEGATIVE_COLOR
            for _, row in df.iterrows()
        ]

        fig.add_trace(
            go.Bar(
                x=df["date"],
                y=df["volume"],
                name="Volume",
                marker_color=colors,
                opacity=0.5
            ),
            row=2,
            col=1
        )

    # Add signals if provided
    if signals:
        buy_signals = [s for s in signals if s.get("direction") == "BUY"]
        sell_signals = [s for s in signals if s.get("direction") == "SELL"]

        if buy_signals:
            fig.add_trace(
                go.Scatter(
                    x=[s.get("date") or s.get("timestamp") for s in buy_signals],
                    y=[s.get("entry_price", s.get("price")) for s in buy_signals],
                    mode="markers",
                    marker=dict(
                        symbol="triangle-up",
                        size=15,
                        color=ChartTheme.POSITIVE_COLOR,
                        line=dict(width=2, color="white")
                    ),
                    name="Buy Signal"
                ),
                row=1 if show_volume else None,
                col=1 if show_volume else None
            )

        if sell_signals:
            fig.add_trace(
                go.Scatter(
                    x=[s.get("date") or s.get("timestamp") for s in sell_signals],
                    y=[s.get("entry_price", s.get("price")) for s in sell_signals],
                    mode="markers",
                    marker=dict(
                        symbol="triangle-down",
                        size=15,
                        color=ChartTheme.NEGATIVE_COLOR,
                        line=dict(width=2, color="white")
                    ),
                    name="Sell Signal"
                ),
                row=1 if show_volume else None,
                col=1 if show_volume else None
            )

    # Apply theme
    layout = ChartTheme.get_layout(f"{ticker} Price Chart", height=600 if show_volume else 400)
    layout["xaxis_rangeslider_visible"] = False

    if show_volume:
        layout["yaxis2"] = {
            "gridcolor": ChartTheme.GRID_COLOR,
            "tickfont": {"size": ChartTheme.TICK_SIZE}
        }

    fig.update_layout(**layout)

    return fig


def create_equity_curve(
    df: pd.DataFrame,
    title: str = "Portfolio Equity Curve",
    show_drawdown: bool = True
) -> go.Figure:
    """
    Create an equity curve chart with optional drawdown.

    Args:
        df: DataFrame with 'date' and 'equity' columns
        title: Chart title
        show_drawdown: Whether to show drawdown subplot

    Returns:
        Plotly Figure
    """
    if show_drawdown:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.7, 0.3],
            subplot_titles=("Equity", "Drawdown")
        )
    else:
        fig = go.Figure()

    # Equity line
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["equity"],
            mode="lines",
            name="Portfolio Value",
            line=dict(color=ChartTheme.PRIMARY_COLOR, width=2),
            fill="tozeroy",
            fillcolor="rgba(52, 152, 219, 0.1)"
        ),
        row=1 if show_drawdown else None,
        col=1 if show_drawdown else None
    )

    # Drawdown
    if show_drawdown and "drawdown" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["drawdown"] * 100,
                mode="lines",
                name="Drawdown",
                line=dict(color=ChartTheme.NEGATIVE_COLOR, width=1.5),
                fill="tozeroy",
                fillcolor="rgba(231, 76, 60, 0.2)"
            ),
            row=2,
            col=1
        )

        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)

    # Apply theme
    layout = ChartTheme.get_layout(title, height=500 if show_drawdown else 350)
    layout["yaxis_title"] = "Portfolio Value ($)"

    fig.update_layout(**layout)

    return fig


def create_performance_chart(
    strategy_data: pd.DataFrame,
    metric: str = "total_pnl"
) -> go.Figure:
    """
    Create a bar chart comparing strategy performance.

    Args:
        strategy_data: DataFrame with strategy performance metrics
        metric: Which metric to display

    Returns:
        Plotly Figure
    """
    fig = go.Figure()

    colors = [
        ChartTheme.POSITIVE_COLOR if val >= 0 else ChartTheme.NEGATIVE_COLOR
        for val in strategy_data[metric]
    ]

    fig.add_trace(
        go.Bar(
            x=strategy_data["strategy"],
            y=strategy_data[metric],
            marker_color=colors,
            text=[f"${v:,.0f}" if metric == "total_pnl" else f"{v:.1f}%"
                  for v in strategy_data[metric]],
            textposition="outside"
        )
    )

    title = metric.replace("_", " ").title()
    layout = ChartTheme.get_layout(f"Strategy {title}", height=400)
    layout["xaxis_title"] = "Strategy"
    layout["yaxis_title"] = title

    fig.update_layout(**layout)

    return fig


def create_win_rate_chart(strategy_data: pd.DataFrame) -> go.Figure:
    """Create a win rate comparison chart."""
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            name="Wins",
            x=strategy_data["strategy"],
            y=strategy_data["wins"],
            marker_color=ChartTheme.POSITIVE_COLOR
        )
    )

    fig.add_trace(
        go.Bar(
            name="Losses",
            x=strategy_data["strategy"],
            y=strategy_data["losses"],
            marker_color=ChartTheme.NEGATIVE_COLOR
        )
    )

    layout = ChartTheme.get_layout("Strategy Win/Loss", height=400)
    layout["barmode"] = "stack"
    layout["xaxis_title"] = "Strategy"
    layout["yaxis_title"] = "Number of Trades"

    fig.update_layout(**layout)

    return fig


def create_pnl_distribution(trades_df: pd.DataFrame) -> go.Figure:
    """Create a P&L distribution histogram."""
    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=trades_df["pnl"],
            nbinsx=30,
            marker_color=ChartTheme.PRIMARY_COLOR,
            opacity=0.75
        )
    )

    # Add vertical line at zero
    fig.add_vline(
        x=0,
        line_dash="dash",
        line_color=ChartTheme.TEXT_COLOR,
        line_width=2
    )

    layout = ChartTheme.get_layout("P&L Distribution", height=350)
    layout["xaxis_title"] = "P&L ($)"
    layout["yaxis_title"] = "Frequency"

    fig.update_layout(**layout)

    return fig


def create_sector_allocation(positions_df: pd.DataFrame, universe_df: pd.DataFrame) -> go.Figure:
    """Create a pie chart of sector allocation."""
    # Merge with universe to get sectors
    merged = positions_df.merge(
        universe_df[["ticker", "sector"]],
        on="ticker",
        how="left"
    )

    sector_counts = merged["sector"].value_counts()

    fig = go.Figure()

    fig.add_trace(
        go.Pie(
            labels=sector_counts.index,
            values=sector_counts.values,
            hole=0.4,
            marker=dict(
                colors=[
                    ChartTheme.PRIMARY_COLOR,
                    ChartTheme.SECONDARY_COLOR,
                    ChartTheme.POSITIVE_COLOR,
                    ChartTheme.NEGATIVE_COLOR,
                    ChartTheme.NEUTRAL_COLOR,
                ] * 3  # Repeat colors
            ),
            textposition="outside",
            textinfo="label+percent"
        )
    )

    layout = ChartTheme.get_layout("Sector Allocation", height=400)
    fig.update_layout(**layout)

    return fig
