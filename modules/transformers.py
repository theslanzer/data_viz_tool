from __future__ import annotations
import pandas as pd
import numpy as np
import ast

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]
    df = df.drop_duplicates()
    for c in df.columns:
        if pd.api.types.is_string_dtype(df[c]):
            df[c] = df[c].str.strip()
    return df

def split_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expect df['label'] to be a JSON-like string convertible with ast.literal_eval
    into a 2-element list/tuple: [labelType, labelName] (or dict with those keys).
    Also computes `lift` from rateWhenPresent / rateWhenAbsent if present.
    """
    out = df.copy()

    # Parse label column: try list/tuple first, fallback to dict
    parsed = out['label'].apply(ast.literal_eval)
    as_df = parsed.apply(pd.Series)
    if {0, 1}.issubset(as_df.columns):
        out[['labelType', 'labelName']] = as_df[[0, 1]]
    elif {'labelType', 'labelName'}.issubset(as_df.columns):
        out[['labelType', 'labelName']] = as_df[['labelType', 'labelName']]
    else:
        # Best-effort: first two columns
        out[['labelType', 'labelName']] = as_df.iloc[:, :2]

    # Compute lift if possible
    if {'rateWhenPresent', 'rateWhenAbsent'}.issubset(out.columns):
        denom = out['rateWhenAbsent'].replace(0, pd.NA)
        out['lift'] = (((out['rateWhenPresent'] - out['rateWhenAbsent']) / denom) * 100).round(0)

    # Normalize labelType names
    label_dict = {
        'Noun Phrase': 'Phrase',
        'video_label': 'Video',
        'hook_text_label': 'Hook',
        'image_label': 'Image',
        'headline_label': 'Headline',
    }
    out['labelType'] = out['labelType'].replace(label_dict)

    # Keep stable order by lift if present
    if 'lift' in out.columns and pd.api.types.is_numeric_dtype(out['lift']):
        out = out.sort_values(by='lift', ascending=False)
    return out

def ensure_labels(df: pd.DataFrame) -> pd.DataFrame:
    if 'labelType' in df.columns and 'labelName' in df.columns:
        # compute lift if missing but rates exist
        if 'lift' not in df.columns and {'rateWhenPresent', 'rateWhenAbsent'}.issubset(df.columns):
            out = df.copy()
            denom = out['rateWhenAbsent'].replace(0, pd.NA)
            out['lift'] = (((out['rateWhenPresent'] - out['rateWhenAbsent']) / denom) * 100).round(0)
            return out
        return df
    if 'label' in df.columns:
        return split_label(df)
    return df

def filter_by_label_type(df: pd.DataFrame, label_type_col: str | None, selected_values: list | None) -> pd.DataFrame:
    if not label_type_col or not selected_values:
        return df
    if label_type_col not in df.columns:
        return df
    return df[df[label_type_col].isin(selected_values)].copy()

def apply_transformations(
    df: pd.DataFrame,
    run_basic_clean: bool,
    label_type_col: str | None,
    label_type_vals: list | None,
) -> pd.DataFrame:
    out = df.copy()
    if run_basic_clean:
        out = basic_clean(out)
    out = ensure_labels(out)
    # optional immediate labelType filtering if you ever enable it
    if label_type_col and label_type_vals:
        out = filter_by_label_type(out, label_type_col, label_type_vals)
    return out

def take_top_n(df: pd.DataFrame, n: int, sort_col: str | None = None, ascending: bool = False) -> pd.DataFrame:
    """
    Keep global Top-N by `sort_col` if provided (numeric), else first N rows (stable).
    """
    n = max(1, int(n))
    out = df.copy()
    if sort_col and sort_col in out.columns and pd.api.types.is_numeric_dtype(out[sort_col]):
        out = out.sort_values(by=sort_col, ascending=ascending)
    return out.head(n)

def top_labels(
    df: pd.DataFrame,
    types: list[str],
    num: int,
    sort_col: str = "lift",
    per_type: bool = False,  # <-- default to combined
) -> pd.DataFrame:
    """
    Return Top-N rows given selected label types.

    - If per_type=True: take top N *per labelType* (old behavior).
    - If per_type=False: take top N *globally across* the selected label types (NEW default).
    """
    if not types:
        # No types selected → just take top-N globally
        sub = df.copy()
    else:
        sub = df[df.get("labelType").isin(types)].copy()

    # Sort by sort_col if numeric; else keep stable order
    if sort_col in sub.columns and pd.api.types.is_numeric_dtype(sub[sort_col]):
        sub = sub.sort_values(by=sort_col, ascending=False, kind="mergesort")

    if per_type:
        # Old behavior: top N for each labelType, then concat
        if "labelType" not in sub.columns:
            return sub.head(num).reset_index(drop=True)
        out = (
            sub.groupby("labelType", group_keys=True, sort=False)
               .head(max(1, int(num)))
               .reset_index(drop=True)
        )
        return out

    # NEW default: Top-N across all selected types combined
    return sub.head(max(1, int(num))).reset_index(drop=True)


def aggregate_lift_by_labeltype(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate present/absent numerators & denominators by labelType,
    then compute lift %.

    Requires columns:
      - labelType
      - numeratorWhenPresent, denominatorWhenPresent
      - numeratorWhenAbsent,  denominatorWhenAbsent
    """
    required = {
        "labelType",
        "numeratorWhenPresent", "denominatorWhenPresent",
        "numeratorWhenAbsent",  "denominatorWhenAbsent",
    }
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"aggregate_lift_by_labeltype: missing columns: {missing}")

    g = df.groupby("labelType", dropna=True).agg(
        numeratorWhenPresent=("numeratorWhenPresent", "sum"),
        denominatorWhenPresent=("denominatorWhenPresent", "sum"),
        numeratorWhenAbsent=("numeratorWhenAbsent", "sum"),
        denominatorWhenAbsent=("denominatorWhenAbsent", "sum"),
    ).reset_index()

    # Avoid divide-by-zero
    pres_rate = g["numeratorWhenPresent"] / g["denominatorWhenPresent"].replace(0, np.nan)
    abs_rate  = g["numeratorWhenAbsent"]  / g["denominatorWhenAbsent"].replace(0, np.nan)

    lift = ((pres_rate - abs_rate) / abs_rate) * 100.0
    g["lift"] = np.round(lift, 0)
    return g.dropna(subset=["lift"]).sort_values("lift", ascending=False).reset_index(drop=True)