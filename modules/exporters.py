from __future__ import annotations
from typing import Any, Dict, Tuple, List
import io
from textwrap import wrap

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

from wordcloud import WordCloud


def resolve_font_path() -> str | None:
    """Return a filesystem path for a font with broad Unicode coverage."""
    try:
        from pathlib import Path as _Path
    except ImportError:  # pragma: no cover
        return None

    candidates = [
        _Path("C:/Windows/Fonts/arialuni.ttf"),
        _Path("C:/Windows/Fonts/ARIALUNI.TTF"),
        _Path("C:/Windows/Fonts/segoeui.ttf"),
        _Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
        _Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        _Path("/Library/Fonts/Arial Unicode.ttf"),
        _Path("/Library/Fonts/Arial.ttf"),
        _Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        _Path("/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"),
        _Path("/usr/share/fonts/truetype/freefont/FreeSans.ttf"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)

    try:
        from matplotlib import font_manager  # type: ignore
        for family in ("Arial Unicode MS", "Segoe UI", "Noto Sans", "DejaVu Sans", "Arial"):
            try:
                path_found = font_manager.findfont(family, fallback_to_default=False)
                if path_found:
                    return path_found
            except Exception:
                continue
    except Exception:
        pass

    return None


def _normalize_sizes(values: np.ndarray, size_range: Tuple[float, float]) -> np.ndarray:
    low, high = size_range
    low = float(low)
    high = float(high)
    if np.isclose(high, low):
        high = low + 1.0
    if values.size == 0:
        return np.zeros_like(values)
    vmin = np.min(values)
    vmax = np.max(values)
    if np.isclose(vmax, vmin):
        return np.full_like(values, high)
    return ( (values - vmin) / (vmax - vmin) ) * (high - low) + low


def generate_wordcloud_png(
    df: pd.DataFrame,
    label_col: str = 'labelName',
    value_col: str = 'lift',
    palette_hexes: List[str] | None = None,
    *,
    width: int = 1100,
    height: int = 650,
    size_range: Tuple[float, float] = (5.0, 40.0),
    prefer_horizontal: float = 0.9,
    background_color: str = 'white',
) -> Tuple[WordCloud, bytes]:
    """Generate a word cloud PNG based on the supplied dataframe."""
    if df.empty:
        raise ValueError('Word cloud dataframe is empty')

    data = df[[label_col, value_col]].dropna().copy()
    if data.empty:
        raise ValueError('No valid rows after dropping NA values')

    labels = data[label_col].astype(str).tolist()
    values = data[value_col].astype(float).to_numpy()
    positive = np.clip(values, a_min=0.0, a_max=None)
    log_vals = np.log1p(positive)
    sizes = _normalize_sizes(log_vals, size_range)
    freq_map = {word: size for word, size in zip(labels, sizes)}

    sorted_pairs = sorted(zip(labels, values), key=lambda x: x[1], reverse=True)
    palette = (palette_hexes or ["#1f2937", "#4338ca", "#f97316"])
    color_lookup: Dict[str, str] = {}
    for idx, (word, _) in enumerate(sorted_pairs):
        color_lookup[word] = palette[idx % len(palette)]

    font_path = resolve_font_path()

    wc = WordCloud(
        width=int(width),
        height=int(height),
        background_color=background_color,
        prefer_horizontal=max(0.0, min(1.0, float(prefer_horizontal))),
        margin=max(1, int((size_range[1] - size_range[0]) // 20 or 1)),
        random_state=42,
        collocations=False,
        normalize_plurals=False,
        font_path=font_path,
        max_words=len(freq_map),
    ).generate_from_frequencies(freq_map)

    def _color_func(word: str, *args: Any, **kwargs: Any) -> str:
        return color_lookup.get(word, palette[0])

    wc = wc.recolor(color_func=_color_func)

    fig = plt.figure(figsize=(width / 100.0, height / 100.0), dpi=120)
    try:
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        fig.tight_layout(pad=0)
        buffer = io.BytesIO()
        fig.savefig(buffer, format="PNG", facecolor=background_color, bbox_inches="tight", pad_inches=0)
        buffer.seek(0)
        png_bytes = buffer.getvalue()
    finally:
        plt.close(fig)

    return wc, png_bytes


def generate_circular_bar_chart_png(
    df: pd.DataFrame,
    label_col: str = "labelName",
    lift_col: str = "lift",
    colors: List[str] | None = None,
    label_wrap: int = 20,
    theta_offset_rad: float = (.7 * np.pi / 2),
    add_margin: float = 50.0,
    inner_frac: float = 0.30,
    show_spokes: bool = True,
    spoke_dash: str = "dash",
    width: int = 900,
    height: int = 900,
) -> bytes:
    """Render a circular bar chart to PNG bytes using Matplotlib."""
    data = df[[label_col, lift_col]].dropna().copy()
    if data.empty:
        raise ValueError("Circular bar chart dataframe is empty")

    data = data.sort_values(by=lift_col, ascending=False).reset_index(drop=True)
    angles = np.linspace(0.0, 2.0 * np.pi, len(data), endpoint=False)
    angles = (angles + float(theta_offset_rad)) % (2.0 * np.pi)
    lengths = data[lift_col].astype(float).to_numpy()
    bar_width = (2.0 * np.pi / max(len(data), 1)) * 0.9

    r_max = float(np.nanmax(lengths)) if lengths.size else 0.0
    r_min = float(np.nanmin(lengths)) if lengths.size else 0.0
    radial_ticks = np.linspace(r_min, r_max, 4)
    if np.allclose(radial_ticks, radial_ticks[0]):
        radial_ticks = np.linspace(radial_ticks[0], radial_ticks[0] + 1.0, 4)
    radial_ticks = np.unique(np.round(radial_ticks, 0))

    if colors and len(colors) >= len(data):
        bar_colors = list(reversed(colors[: len(data)]))
    else:
        cmap = plt.get_cmap("viridis")
        norm = mcolors.Normalize(
            vmin=r_min,
            vmax=r_max if not np.isclose(r_max, r_min) else r_min + 1.0,
        )
        source_values = lengths if lengths.size else np.array([0.0])
        bar_colors = cmap(norm(source_values))

    outer_radius = r_max + float(add_margin)
    inner_radius = -abs(r_max) * float(inner_frac)

    fig = plt.figure(figsize=(max(width, 200) / 100.0, max(height, 200) / 100.0), dpi=100)
    ax = fig.add_subplot(111, projection="polar")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.set_theta_offset(float(theta_offset_rad))
    ax.set_ylim(inner_radius, outer_radius)

    ax.bar(angles, lengths, color=bar_colors, width=bar_width, alpha=0.9, zorder=10)

    if show_spokes:
        linestyle = (0, (4, 4)) if spoke_dash == "dash" else "solid"
        ax.vlines(angles, 0, r_max, colors="#1f1f1f", linestyles=linestyle, linewidth=1.0, zorder=11)
    ax.xaxis.grid(False)

    if label_wrap and label_wrap > 0:
        xtick_labels = ["\n".join(wrap(str(label), label_wrap)) for label in data[label_col]]
    else:
        xtick_labels = data[label_col].astype(str).tolist()
    ax.set_xticks(angles)
    ax.set_xticklabels(xtick_labels, size=12)

    ax.set_yticks(radial_ticks)
    ax.set_yticklabels([])

    ax.spines["start"].set_color("none")
    ax.spines["polar"].set_color("none")
    for tick in ax.xaxis.get_major_ticks():
        tick.set_pad(10)

    pad = 10.0
    annotation_angle = float(theta_offset_rad) - (0.2 * np.pi / 2.0)
    for val in radial_ticks:
        ax.text(annotation_angle, float(val) + pad, f"{int(val)}", ha="center", size=10)

    # fig.tight_layout()
    buffer = io.BytesIO()
    try:
        fig.savefig(buffer, format="PNG", facecolor="white", bbox_inches="tight", pad_inches=0.1)
        buffer.seek(0)
        png_bytes = buffer.getvalue()
    finally:
        plt.close(fig)

    return png_bytes
