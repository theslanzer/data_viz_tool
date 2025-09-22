# modules/charts.py
from __future__ import annotations
import numpy as np
import pandas as pd
import textwrap
import plotly.graph_objects as go
import random

def _wrap_labels(labels, width=20):
    return ["\n".join(textwrap.wrap(str(x), width=width)) for x in labels]

def circular_bar_interactive(
    df: pd.DataFrame,
    label_col: str = "labelName",
    lift_col: str = "lift",
    colors: list[str] | None = None,
    label_wrap: int = 20,
    theta_offset_rad: float = (1.2 * np.pi / 2),  # match your matplotlib offset
    add_margin: float = 50.0,                     # extra headroom like +50 in ylim
    inner_frac: float = 0.30,                     # inner negative radius as a fraction of max
    show_spokes: bool = True,
    spoke_dash: str = "dash",                     # 'dash' to mimic (0,(4,4))
    width: int = 900,           # new: chart width in px
    height: int = 900,          # new: chart height in px
) -> go.Figure:
    """
    Interactive circular bar chart mimicking your matplotlib styling.

    - Bars positioned on a polar axis with same theta offset
    - Colors mapped to lift (list of hex provided by caller)
    - Wrapped xtick labels, dashed spokes, extended radial range
    """
    data = df[[label_col, lift_col]].dropna().copy()
    data = data.sort_values(by=lift_col).reset_index(drop=True)
    if data.empty:
        # Return an empty but valid figure to avoid crashes
        fig = go.Figure()
        fig.update_layout(title="No data to display")
        return fig

    n = len(data)
    # Angles (degrees)
    theta = np.linspace(0, 360, n, endpoint=False)

    # Apply the same theta offset you used in Matplotlib
    theta_offset_deg = np.degrees(theta_offset_rad)
    theta = (theta + theta_offset_deg) % 360

    # Radial range to mimic: from - (max*inner_frac) to max + add_margin
    r_vals = data[lift_col].astype(float).to_numpy()
    r_max = np.nanmax(r_vals)
    r_min = -float(r_max) * float(inner_frac)
    r_hi  = float(r_max) + float(add_margin)

    # Colors (discrete list sized to n)
    if colors is None or len(colors) < n:
        # Fallback neutral gradient
        colors = ["#a3a3a3"] * n
    bar_colors = colors[:n]

    # Build base barpolar
    fig = go.Figure(go.Barpolar(
        r=r_vals,
        theta=theta,
        text=data[label_col].astype(str).tolist(),
        hovertemplate="<b>%{text}</b><br>Lift: %{r:.0f}%<extra></extra>",
        marker=dict(
            color=bar_colors,
            line=dict(width=1, color="rgba(31,31,31,0.4)")
        ),
        opacity=0.9
    ))

    # Optional dashed spokes like ax.vlines(...)
    if show_spokes:
        # add one Scatterpolar per spoke
        for ang in theta:
            fig.add_trace(go.Scatterpolar(
                r=[0, r_max],
                theta=[ang, ang],
                mode="lines",
                line=dict(color="rgba(31,31,31,0.5)", width=1, dash=spoke_dash),
                hoverinfo="skip",
                showlegend=False
            ))

    # Wrapped tick labels
    ticktext = _wrap_labels(data[label_col].tolist(), width=label_wrap)

    # Build 4 y ticks like your labels = linspace(min,max,4).astype(int)
    ytick_vals = np.linspace(np.nanmin(r_vals), r_max, 4)
    ytick_vals = np.unique(np.round(ytick_vals, 0)).tolist()

    fig.update_layout(
        template="plotly",
        showlegend=False,
        width=width,
        height=height,
        margin=dict(l=40, r=40, t=40, b=40),
        polar=dict(
            bgcolor="white",
            radialaxis=dict(
                range=[r_min, r_hi],
                showline=False,
                gridcolor="rgba(0,0,0,0.15)",
                tickmode="array",
                tickvals=ytick_vals,
                ticktext=[f"{int(v)}" for v in ytick_vals],
                tickfont=dict(size=14, color ='grey'),
                ticks="",  # hide radial tick marks
            ),
            angularaxis=dict(
                direction="clockwise", # match your matplotlib
                rotation=210,               # 0° at top
                tickmode="array",
                tickvals=theta,           # positions
                ticktext=ticktext,        # wrapped labels
                tickfont=dict(size=16),
                gridcolor="rgba(0,0,0,0)",# we draw our own spokes
            ),
        ),
    )
    return fig

def interactive_wordcloud(
    df: pd.DataFrame,
    label_col: str = "labelName",
    lift_col: str = "lift",
    palette_hexes: list[str] | None = None,
    width: int = 1400,
    height: int = 800,
    min_font_px: int = 10,
    max_font_px: int = 48,
    max_words: int = 200,
    attempts_per_word: int = 2000,
    padding_px: float = 8.0,
) -> go.Figure:
    """
    Plotly-based word cloud using random placement with collision detection.
    """

    data = df[[label_col, lift_col]].dropna().copy()
    if data.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data to display", showarrow=False,
                           x=0.5, y=0.5, xref="paper", yref="paper")
        fig.update_layout(width=width, height=height)
        return fig

    # Aggregate duplicates
    data = data.groupby(label_col, as_index=False)[lift_col].max()
    data = data.sort_values(lift_col, ascending=False).head(max_words).reset_index(drop=True)

    # Scale sizes
    vals = data[lift_col].to_numpy(float)
    vals_log = np.log1p(vals)
    lo, hi = vals_log.min(), vals_log.max()
    if hi - lo < 1e-12:
        sizes = np.full(vals_log.shape, (min_font_px+max_font_px)/2.0)
    else:
        sizes = min_font_px + (vals_log - lo)/(hi-lo) * (max_font_px - min_font_px)

    # Colors: simple gradient (reuse your generate_palette if available)
    from modules.colors import generate_palette
    palette = generate_palette(",".join(palette_hexes or ["#6b21a8", "#c026d3"]), n=len(vals))
    colors = palette

    # Bounding box helper
    def bbox(word, fs, cx, cy):
        w = 0.75 * fs * max(1, len(word))
        h = 1.20 * fs
        return (cx - w/2 - padding_px, cy - h/2 - padding_px,
                cx + w/2 + padding_px, cy + h/2 + padding_px)

    def intersects(a, b):
        return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])

    placed = []
    for (word, val, fs, col) in zip(data[label_col].astype(str), vals, sizes, colors):
        ok = False
        for _ in range(attempts_per_word):
            x = random.uniform(0, width)
            y = random.uniform(0, height)
            box = bbox(word, fs, x, y)

            # Stay inside canvas
            if box[0] < 0 or box[1] < 0 or box[2] > width or box[3] > height:
                continue

            # Check overlaps
            if all(not intersects(box, p[-1]) for p in placed):
                placed.append((x, y, fs, col, word, val, box))
                ok = True
                break

        if not ok:
            # word skipped if no valid spot found
            continue

    if not placed:
        fig = go.Figure()
        fig.add_annotation(text="No words could be placed.", showarrow=False,
                           x=0.5, y=0.5, xref="paper", yref="paper")
        fig.update_layout(width=width, height=height)
        return fig

    xs  = [p[0] for p in placed]
    ys  = [p[1] for p in placed]
    fns = [p[2] for p in placed]
    cls = [p[3] for p in placed]
    wrd = [p[4] for p in placed]
    lft = [p[5] for p in placed]

    default_fs = 12.0  # or float(min_font_px) if available in this scope
    clean_fns = []
    for v in fns:
        try:
            fv = float(v)
        except Exception:
            fv = default_fs
        if not np.isfinite(fv) or fv < 1:
            fv = max(1.0, default_fs)
        clean_fns.append(fv)
    # -------------------------------------------

    fig = go.Figure(go.Scatter(
        x=xs, y=ys, mode="text",
        text=wrd,
        textfont=dict(
            size=clean_fns,              # 👈 use the sanitized sizes
            color=cls,
            family="Arial Black"         # (or your chosen family)
        ),
        hovertemplate="<b>%{text}</b><br>Lift: %{customdata:.2f}%<extra></extra>",
        customdata=lft
    ))
    fig.update_xaxes(visible=False, range=[0, width])
    fig.update_yaxes(visible=False, range=[0, height], scaleanchor="x", scaleratio=1)
    fig.update_layout(
        width=width, height=height,
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=10, r=10, t=10, b=10),
        dragmode="pan"
    )
    return fig

def bar_lift_by_type_interactive(
    df: pd.DataFrame,
    label_col: str = "labelType",
    lift_col: str = "lift",
    colors: list[str] | None = None,
    width: int = 1000,
    height: int = 600,
    title: str = "Lift by Label Type",
) -> go.Figure:
    """
    Interactive bar chart; assumes df already aggregated by label type with a `lift` column.
    Bars are sorted by lift desc. If `colors` is provided, it will be used in order.
    """
    data = df[[label_col, lift_col]].dropna().copy()
    if data.empty:
        fig = go.Figure()
        fig.update_layout(title="No data for bar chart", width=width, height=height)
        return fig

    data = data.sort_values(lift_col, ascending=False).reset_index(drop=True)
    bar_colors = colors[: len(data)] if colors else None

    fig = go.Figure(
        go.Bar(
            x=data[label_col],
            y=data[lift_col],
            marker=dict(color=bar_colors) if bar_colors else None,
            hovertemplate="<b>%{x}</b><br>Lift: %{y:.0f}%<extra></extra>",
        )
    )
    fig.update_layout(
        width=width,
        height=height,
        title=title,
        xaxis=dict(
            title="Label Types", 
            # titlefont=dict(color="black", size=16),
            tickangle=0, 
            tickfont=dict(color="black", size=14)
            ),
        yaxis=dict(
            title="Lift (%)", 
            # titlefont=dict(color="black", size=16),
            zeroline=True, 
            zerolinewidth=1, 
            gridcolor="rgba(0,0,0,0.1)",
            tickfont=dict(color="grey", size=14)
            ),
        margin=dict(l=40, r=20, t=50, b=60),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig
