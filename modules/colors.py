# modules/colors.py
from __future__ import annotations
from typing import List, Tuple

def _normalize_hex(h: str) -> str:
    h = h.strip()
    if not h:
        return h
    if h[0] != "#":
        h = "#" + h
    if len(h) == 4:  # #abc -> #aabbcc
        h = "#" + "".join([ch*2 for ch in h[1:]])
    return h.lower()

def _hex_to_rgb(h: str) -> Tuple[int, int, int]:
    h = _normalize_hex(h)
    if len(h) != 7:
        raise ValueError(f"Invalid hex color: {h}")
    return tuple(int(h[i:i+2], 16) for i in (1, 3, 5))

def _rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    r, g, b = rgb
    return "#{:02x}{:02x}{:02x}".format(
        max(0, min(255, r)),
        max(0, min(255, g)),
        max(0, min(255, b))
    )

def _blend(c1: Tuple[int,int,int], c2: Tuple[int,int,int], t: float) -> Tuple[int,int,int]:
    """Linear blend: t=0 -> c1, t=1 -> c2"""
    r = round(c1[0] + (c2[0]-c1[0]) * t)
    g = round(c1[1] + (c2[1]-c1[1]) * t)
    b = round(c1[2] + (c2[2]-c1[2]) * t)
    return (r, g, b)

def parse_hex_list(raw: str) -> List[str]:
    """
    Accept comma or space separated hex codes.
    """
    if not raw:
        return []
    parts = [p for chunk in raw.split(",") for p in chunk.split()]
    return [_normalize_hex(p) for p in parts if p.strip()]

def generate_palette(raw_hex: str, n: int) -> List[str]:
    """
    If one hex provided: gradient from light version -> hex.
    If multiple: build a multi-stop gradient across them.
    Always returns length n (repeats last color if n < 1).
    """
    n = max(1, int(n))
    hexes = parse_hex_list(raw_hex)

    if not hexes:
        # default neutral gradient
        hexes = ["#a3a3a3", "#1f2937"]  # gray-400 -> gray-800

    rgbs = [_hex_to_rgb(h) for h in hexes]

    # One color: blend from white to that color
    if len(rgbs) == 1:
        start = (255, 255, 255)
        end   = rgbs[0]
        return [_rgb_to_hex(_blend(start, end, t)) for t in [i/(n-1) if n>1 else 1 for i in range(n)]]

    # Multi-stop gradient
    stops = rgbs
    segments = len(stops) - 1
    if n == 1:
        return [_rgb_to_hex(stops[-1])]

    out: List[str] = []
    # Distribute n colors across segments
    per = [n // segments] * segments
    for i in range(n % segments):
        per[i] += 1

    for seg_idx in range(segments):
        c1, c2 = stops[seg_idx], stops[seg_idx+1]
        k = per[seg_idx]
        if seg_idx == segments - 1:
            # include last color at end
            t_vals = [i/(k-1) if k>1 else 1 for i in range(k)]
        else:
            # avoid duplicating boundary colors
            t_vals = [i/k for i in range(k)]
        for t in t_vals:
            out.append(_rgb_to_hex(_blend(c1, c2, t)))

    # Safety
    if len(out) < n:
        out += [out[-1]] * (n - len(out))
    return out[:n]
