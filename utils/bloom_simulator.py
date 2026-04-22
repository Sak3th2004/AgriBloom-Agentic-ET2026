"""
Bloom Simulator - Animated Crop Health Trajectory Visualization
Shows "before vs after" treatment health projections using Plotly
"""
from __future__ import annotations

import math
from typing import Any

import plotly.graph_objects as go


def _calculate_recovery_curve(
    start: float,
    end: float,
    days: int,
    curve_type: str = "sigmoid",
) -> list[float]:
    """
    Calculate recovery curve based on treatment effectiveness.

    Args:
        start: Starting health score (0-100)
        end: Target health score (0-100)
        days: Number of days to project
        curve_type: "linear", "sigmoid", or "exponential"

    Returns:
        List of health scores for each day
    """
    if curve_type == "linear":
        return [start + (end - start) * i / max(1, days - 1) for i in range(days)]

    elif curve_type == "sigmoid":
        # S-curve for realistic disease recovery
        values = []
        for i in range(days):
            # Sigmoid function centered at day 7
            x = (i - days / 2) / (days / 4)
            sigmoid = 1 / (1 + math.exp(-x))
            value = start + (end - start) * sigmoid
            values.append(value)
        return values

    elif curve_type == "exponential":
        # Exponential decay towards target
        values = []
        decay_rate = -math.log(0.1) / days  # 90% of change in 'days' period
        for i in range(days):
            remaining = (end - start) * (1 - math.exp(-decay_rate * i))
            values.append(start + remaining)
        return values

    return [start + (end - start) * i / max(1, days - 1) for i in range(days)]


def _calculate_baseline_curve(start: float, days: int, decline_rate: float = 0.5) -> list[float]:
    """
    Calculate baseline (untreated) curve showing slow decline.

    Args:
        start: Starting health score
        days: Number of days
        decline_rate: Daily decline rate (0-2)

    Returns:
        List of health scores showing gradual decline
    """
    values = []
    for i in range(days):
        # Slight improvement due to natural resilience, then plateau
        natural_recovery = min(5, i * 0.3)
        decline = i * decline_rate * 0.3
        value = start + natural_recovery - decline
        values.append(max(10, min(100, value)))
    return values


def build_bloom_figure(
    before_health: float,
    after_health: float,
    days: int = 14,
    disease_severity: str = "medium",
    crop_name: str = "Crop",
) -> go.Figure:
    """
    Build an animated Plotly figure showing crop health trajectory.

    Args:
        before_health: Current health score (0-100)
        after_health: Projected health after treatment (0-100)
        days: Projection period in days
        disease_severity: "low", "medium", "high", "critical"
        crop_name: Name of the crop for display

    Returns:
        Plotly Figure object with animated chart
    """
    # Validate inputs
    before_health = max(5.0, min(100.0, before_health))
    after_health = max(before_health, min(100.0, after_health))
    days = max(7, min(30, days))

    # Determine curve parameters based on severity
    severity_config = {
        "low": {"curve": "exponential", "decline": 0.2, "color": "#4ade80"},
        "medium": {"curve": "sigmoid", "decline": 0.5, "color": "#22c55e"},
        "high": {"curve": "sigmoid", "decline": 0.8, "color": "#16a34a"},
        "critical": {"curve": "linear", "decline": 1.2, "color": "#15803d"},
    }
    config = severity_config.get(disease_severity, severity_config["medium"])

    # Calculate curves
    x_days = list(range(1, days + 1))
    baseline = _calculate_baseline_curve(before_health, days, config["decline"])
    improved = _calculate_recovery_curve(before_health, after_health, days, config["curve"])

    # Create figure
    fig = go.Figure()

    # Baseline (untreated) curve
    fig.add_trace(
        go.Scatter(
            x=x_days,
            y=baseline,
            mode="lines+markers",
            name="Without Treatment",
            line={"color": "#f59e0b", "width": 3, "dash": "dot"},
            marker={"size": 6, "symbol": "circle"},
            hovertemplate="Day %{x}<br>Health: %{y:.1f}%<extra>Without Treatment</extra>",
        )
    )

    # Improved (treated) curve with fill
    fig.add_trace(
        go.Scatter(
            x=x_days,
            y=improved,
            mode="lines+markers",
            name="With AgriBloom Plan",
            line={"color": config["color"], "width": 4},
            marker={"size": 8, "symbol": "diamond"},
            fill="tozeroy",
            fillcolor=f"rgba(34, 197, 94, 0.15)",
            hovertemplate="Day %{x}<br>Health: %{y:.1f}%<extra>With Treatment</extra>",
        )
    )

    # Add improvement annotation
    improvement = after_health - before_health
    fig.add_annotation(
        x=days,
        y=improved[-1],
        text=f"+{improvement:.0f}%",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="#22c55e",
        font={"size": 14, "color": "#dcfce7", "family": "Arial Black"},
        bgcolor="rgba(22, 101, 52, 0.8)",
        bordercolor="#22c55e",
        borderwidth=2,
        borderpad=4,
    )

    # Add threshold lines
    fig.add_hline(
        y=80,
        line_dash="dash",
        line_color="rgba(74, 222, 128, 0.5)",
        annotation_text="Healthy",
        annotation_position="right",
        annotation_font_color="#86efac",
    )
    fig.add_hline(
        y=50,
        line_dash="dash",
        line_color="rgba(251, 191, 36, 0.5)",
        annotation_text="At Risk",
        annotation_position="right",
        annotation_font_color="#fde68a",
    )
    fig.add_hline(
        y=25,
        line_dash="dash",
        line_color="rgba(239, 68, 68, 0.5)",
        annotation_text="Critical",
        annotation_position="right",
        annotation_font_color="#fca5a5",
    )

    # Update layout
    fig.update_layout(
        title={
            "text": f"🌱 Bloom Simulator: {crop_name} Recovery Trajectory",
            "font": {"size": 18, "color": "#dcfce7", "family": "Arial"},
            "x": 0.5,
            "xanchor": "center",
        },
        xaxis={
            "title": {"text": "Days After Treatment", "font": {"color": "#86efac"}},
            "tickfont": {"color": "#dcfce7"},
            "gridcolor": "rgba(74, 222, 128, 0.1)",
            "range": [0.5, days + 0.5],
        },
        yaxis={
            "title": {"text": "Crop Health Score (%)", "font": {"color": "#86efac"}},
            "tickfont": {"color": "#dcfce7"},
            "gridcolor": "rgba(74, 222, 128, 0.1)",
            "range": [0, 105],
        },
        template="plotly_dark",
        plot_bgcolor="#052e16",
        paper_bgcolor="#052e16",
        font={"color": "#dcfce7", "family": "Inter, Arial, sans-serif"},
        legend={
            "orientation": "h",
            "y": -0.15,
            "x": 0.5,
            "xanchor": "center",
            "bgcolor": "rgba(5, 46, 22, 0.8)",
            "bordercolor": "rgba(74, 222, 128, 0.3)",
            "borderwidth": 1,
        },
        margin={"l": 60, "r": 30, "t": 80, "b": 80},
        height=400,
    )

    return fig


def build_comparison_figure(
    scenarios: list[dict[str, Any]],
    days: int = 14,
) -> go.Figure:
    """
    Build comparison figure showing multiple treatment scenarios.

    Args:
        scenarios: List of dicts with 'name', 'before', 'after', 'color'
        days: Projection period

    Returns:
        Plotly Figure with multiple scenario comparison
    """
    fig = go.Figure()
    x_days = list(range(1, days + 1))

    for scenario in scenarios:
        improved = _calculate_recovery_curve(
            scenario["before"],
            scenario["after"],
            days,
            "sigmoid",
        )

        fig.add_trace(
            go.Scatter(
                x=x_days,
                y=improved,
                mode="lines+markers",
                name=scenario["name"],
                line={"color": scenario.get("color", "#22c55e"), "width": 3},
                marker={"size": 6},
            )
        )

    fig.update_layout(
        title="Treatment Scenario Comparison",
        xaxis_title="Days",
        yaxis_title="Health Score (%)",
        template="plotly_dark",
        plot_bgcolor="#052e16",
        paper_bgcolor="#052e16",
        font={"color": "#dcfce7"},
    )

    return fig


# Export
__all__ = ["build_bloom_figure", "build_comparison_figure"]
