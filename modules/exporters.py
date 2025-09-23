from __future__ import annotations
from typing import Any, Dict, Tuple, List
import io

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt

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

