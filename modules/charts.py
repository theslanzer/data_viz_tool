# modules/charts.py
from __future__ import annotations
import base64
from typing import Any, Dict, List
import numpy as np
import pandas as pd
import textwrap
import plotly.graph_objects as go

from modules.exporters import generate_wordcloud_png, resolve_font_path

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
    bar_colors = list(reversed(colors[:n]))

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


def _resolve_streamlit_wc_func():
    """Return the callable for the streamlit-wordcloud component if available."""
    import importlib

    module_names = ('streamlit_wordcloud', 'streamlit_wordcloud_v2', 'st_wordcloud')
    attr_names = ('visualize', 'wordcloud', 'st_wordcloud')
    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        for attr in attr_names:
            func = getattr(module, attr, None)
            if callable(func):
                return func
    return None


def wordcloud_v2_component(
    df: pd.DataFrame,
    label_col: str = 'labelName',
    value_col: str = 'lift',
    palette_hexes: List[str] | None = None,
    width: int | str = 1100,
    height: int | str = 650,
    max_words: int = 100,
    font_min: int = 14,
    font_max: int = 60,
    padding: int = 6,
    prefer_horizontal: float = 1.0,
    background_color: str = 'white',
    bold: bool = True,
    tooltip_label: str = 'Lift (%)',
    component_key: str | None = None,
) -> Dict[str, Any] | None:
    import inspect

    data = df[[label_col, value_col]].dropna().copy()
    if data.empty:
        return None

    data = data.groupby(label_col, as_index=False)[value_col].max()
    data = data.sort_values(by=value_col, ascending=False).head(int(max_words)).reset_index(drop=True)
    data = data[data[value_col] > 0]
    if data.empty:
        return None

    palette = list(palette_hexes) if palette_hexes else ['#1f77b4']
    if not palette:
        palette = ['#1f77b4']

    label_values = data[label_col].astype(str).tolist()
    color_map = {word: palette[idx % len(palette)] for idx, word in enumerate(label_values)}
    lift_map = {word: float(val) for word, val in zip(label_values, data[value_col].astype(float))}

    def _coerce_dimension(raw: int | str, fallback: int) -> int:
        if isinstance(raw, (int, float)):
            return max(int(raw), 10)
        if isinstance(raw, str):
            stripped = raw.strip().lower()
            for suffix in ('px',):
                if stripped.endswith(suffix):
                    stripped = stripped[:-len(suffix)]
            digits = ''.join(ch for ch in stripped if (ch.isdigit() or ch == '.'))
            try:
                if digits:
                    return max(int(float(digits)), 10)
            except ValueError:
                pass
        return fallback

    width_px = _coerce_dimension(width, 1100)
    height_px = _coerce_dimension(height, 650)

    try:
        horizontal_ratio = float(prefer_horizontal)
    except (TypeError, ValueError):
        horizontal_ratio = 1.0
    horizontal_ratio = max(0.0, min(1.0, horizontal_ratio)) or 1.0

    size_low = float(font_min) if font_min is not None else 14.0
    size_high = float(font_max) if font_max is not None else 60.0
    wc, image_bytes = generate_wordcloud_png(
        df=data[[label_col, value_col]].rename(columns={label_col: 'labelName', value_col: 'lift'}),
        label_col='labelName',
        value_col='lift',
        palette_hexes=palette,
        width=width_px,
        height=height_px,
        size_range=(size_low, size_high),
        prefer_horizontal=horizontal_ratio,
        background_color=background_color,
    )



    words_payload: List[Dict[str, Any]] = []
    for idx, row in data.iterrows():
        word_text = str(row[label_col])
        if not word_text:
            continue
        lift_value = float(row[value_col])
        item: Dict[str, Any] = {
            'text': word_text,
            'value': lift_value,
            'lift': lift_value,
        }
        if palette_hexes:
            item['color'] = palette_hexes[idx % len(palette_hexes)]
        if bold:
            item['fontWeight'] = 'bold'
        words_payload.append(item)

    if not words_payload:
        return None

    wc_func = _resolve_streamlit_wc_func()
    component_value = None
    if wc_func is not None:
        width_arg = f"{width_px}px"
        height_arg = f"{height_px}px"
        tooltip_fields = {'text': 'Label', 'lift': tooltip_label}
        component_kwargs = {
            'words': words_payload,
            'width': width_arg,
            'height': height_arg,
            'font_min': int(font_min),
            'font_max': int(font_max),
            'max_words': int(max_words),
            'padding': int(padding),
            'layout': 'rectangular',
            'enable_tooltip': True,
            'tooltip_data_fields': tooltip_fields,
            'per_word_coloring': True,
            'ignore_hover': True,
            'ignore_click': True,
            'key': component_key,
            'fontFamily': 'Arial',
            'deterministic': True,
        }
        allowed = set(inspect.signature(wc_func).parameters.keys())
        clean_kwargs = {k: v for k, v in component_kwargs.items() if k in allowed and v is not None}
        component_value = wc_func(**clean_kwargs)

    fallback_fig = None
    if wc_func is None:
        from PIL import ImageFont

        src_width, src_height = wc.width, wc.height
        scale_x = src_width / max(width_px, 1)
        scale_y = src_height / max(height_px, 1)

        effective_font_path = wc.font_path or resolve_font_path()
        font_cache: Dict[int, ImageFont.FreeTypeFont] = {}

        def _measure(word: str, font_size: int) -> tuple[int, int]:
            cache_key = max(font_size, 1)
            if cache_key not in font_cache:
                try:
                    if effective_font_path:
                        font_cache[cache_key] = ImageFont.truetype(effective_font_path, cache_key)
                    else:
                        raise OSError('no font path available')
                except Exception:
                    font_cache[cache_key] = ImageFont.load_default()
            bbox = font_cache[cache_key].getbbox(word)
            width_val = bbox[2] - bbox[0]
            height_val = bbox[3] - bbox[1]
            return max(width_val, 1), max(height_val, 1)
        xs: List[float] = []
        ys: List[float] = []
        words_plot: List[str] = []
        text_sizes: List[int] = []
        text_colors: List[str] = []
        hover_vals: List[float] = []

        for (word, _freq), font_size, (x, y), orientation, color in wc.layout_:
            word_text = str(word)
            if not word_text:
                continue
            base_size = max(int(round(font_size / max(scale_x, 1))), 1)
            text_width, text_height = _measure(word_text, base_size)
            if orientation and orientation != 0:
                text_width, text_height = text_height, text_width
            center_x = (x + text_width / 2) / max(scale_x, 1)
            center_y = (y + text_height / 2) / max(scale_y, 1)
            words_plot.append(word_text)
            xs.append(center_x)
            ys.append(center_y)
            text_sizes.append(base_size)
            text_colors.append(color or color_map.get(word_text, '#1f2937'))
            hover_vals.append(lift_map.get(word_text, 0.0))

        fallback_fig = go.Figure(
            go.Scatter(
                x=xs,
                y=ys,
                mode='text',
                text=words_plot,
                textposition='middle center',
                textfont=dict(
                    family='Arial',
                    size=text_sizes,
                    color=text_colors,
                ),
                customdata=hover_vals,
                hovertemplate=f"<b>%{text}</b><br>{tooltip_label}: %{customdata:.1f}%<extra></extra>",
            )
        )
        fallback_fig.update_layout(
            width=width_px,
            height=height_px,
            margin=dict(l=0, r=0, t=0, b=0),
            plot_bgcolor=background_color,
            paper_bgcolor=background_color,
            hoverlabel=dict(font=dict(family='Arial')),
        )
        fallback_fig.update_xaxes(visible=False, showgrid=False, zeroline=False, range=[0, width_px])
        fallback_fig.update_yaxes(visible=False, showgrid=False, zeroline=False, range=[height_px, 0], scaleanchor='x', scaleratio=1)

    result: Dict[str, Any] = {
        'image_bytes': image_bytes,
        'data': data[[label_col, value_col]].copy(),
    }
    if component_value is not None:
        result['component_value'] = component_value
    if fallback_fig is not None:
        result['figure'] = fallback_fig
    return result


def bar_lift_by_type_interactive(
    df: pd.DataFrame,
    label_col: str = "labelType",
    lift_col: str = "lift",
    colors: list[str] | None = None,
    width: int = 1000,
    height: int = 600,
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
        xaxis=dict(
            title=dict(text="Label Types", font=dict(color="#000000", size=16)),
            tickangle=0, 
            tickfont=dict(color="black", size=14),
            ),
        yaxis=dict(
            title=dict(text="Lift (%)", font=dict(color="#000000", size=16)),
            zeroline=True, 
            zerolinewidth=1, 
            gridcolor="rgba(0,0,0,0.1)",
            tickfont=dict(color="black", size=14)
            ),
        margin=dict(l=40, r=20, t=50, b=60),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig
