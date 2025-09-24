# app.py
from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import streamlit_authenticator as stauth
import json
import inspect
from typing import Any, Dict, List

# --- Local modules ---
from modules.io_utils import read_uploaded_file, SUPPORTED_EXTS
from modules.transformers import apply_transformations, top_labels, aggregate_lift_by_labeltype
from modules.colors import generate_palette
from modules.charts import circular_bar_interactive, wordcloud_v2_component, bar_lift_by_type_interactive
from modules.exporters import generate_circular_bar_chart_png, generate_bar_lift_by_type_png

# -----------------------------
# Login
# -----------------------------

def require_login():
    session_state = st.session_state
    # Make a mutable copy of secrets (Authenticator mutates credentials)
    secrets = st.secrets.to_dict() if hasattr(st.secrets, "to_dict") else json.loads(json.dumps(dict(st.secrets)))
    creds   = secrets.get("credentials", {})
    authcfg = secrets.get("auth", {})

    authenticator = stauth.Authenticate(
        credentials=creds,
        cookie_name=authcfg.get("cookie_name", "viz_builder_auth"),
        key=authcfg.get("cookie_key", "change_me"),
        cookie_expiry_days=int(authcfg.get("cookie_expiry_days", 7)),
        preauthorized=authcfg.get("preauthorized", []),
    )

    login_func = authenticator.login
    login_sig = inspect.signature(login_func)
    login_params = login_sig.parameters
    requires_form_name = (
        "form_name" in login_params
        and login_params["form_name"].default is inspect._empty
    )

    auth_container_left, auth_container_center, auth_container_right = st.columns([2, 1, 2])
    with auth_container_center:
        status_placeholder = st.empty()
        if requires_form_name:
            login_result = login_func("Login", location="main")
        else:
            login_kwargs = {}
            if "form_name" in login_params and login_params["form_name"].default is not inspect._empty:
                login_kwargs["form_name"] = "Login"
            if "location" in login_params:
                login_kwargs["location"] = "main"
            if "max_login_attempts" in login_params:
                login_kwargs["max_login_attempts"] = 3
            if "fields" in login_params:
                login_kwargs["fields"] = {"Form name": "Login"}
            if "key" in login_params:
                login_kwargs.setdefault("key", "main_login_form")
            login_result = login_func(**login_kwargs)
    
    if isinstance(login_result, tuple) and len(login_result) == 3:
        name, auth_status, username = login_result
    else:
        name = session_state.get("name")
        auth_status = session_state.get("authentication_status")
        username = session_state.get("username")

    # Streamlit-authenticator marks logout by setting `logout` flag and flipping auth status to False
    just_logged_out = bool(session_state.pop("logout", False))
    if just_logged_out:
        auth_status = None
        session_state["authentication_status"] = None
        session_state["name"] = None
        session_state["username"] = None

    if auth_status:
        status_placeholder.empty()
        with st.sidebar:
            logout_func = authenticator.logout
            logout_sig = inspect.signature(logout_func)
            logout_params = logout_sig.parameters

            if "button_name" in logout_params or "location" in logout_params:
                logout_kwargs = {}
                if "button_name" in logout_params:
                    logout_kwargs["button_name"] = "Logout"
                if "location" in logout_params:
                    logout_kwargs["location"] = "sidebar"
                logout_func(**logout_kwargs)
            else:
                logout_func("Logout")
        return {"name": name, "username": username}

    status_placeholder.empty()
    if auth_status is False:
        status_placeholder.error("Username/password is incorrect")
    else:
        status_placeholder.warning("Please enter your credentials")
    return None
login_state = require_login()
if login_state is None:
    st.stop()


# -----------------------------
# Helpers
# -----------------------------
def render_fig_grid(items, cols=2):
    """Render charts or render callbacks in a grid with `cols` columns per row."""
    if not items:
        return
    cols = max(1, min(int(cols), 3))
    for i in range(0, len(items), cols):
        row = st.columns(cols, gap="large")
        for j, col in enumerate(row):
            idx = i + j
            if idx >= len(items):
                break
            entry = items[idx]
            if not isinstance(entry, dict):
                entry = {"figure": entry}
            title = entry.get("title") or ""
            with col:
                if title:
                    st.markdown(f"##### {title}")
                figure = entry.get("figure")
                if figure is not None:
                    st.plotly_chart(figure, use_container_width=True)
                else:
                    renderer = entry.get("render")
                    if callable(renderer):
                        renderer()

def _render_labeltype_chips(df: pd.DataFrame):
    if "labelType" not in df.columns:
        return
    present = sorted(df["labelType"].dropna().astype(str).unique().tolist())
    if not present:
        return
    chip_css = """
    <style>
      .chip {display:inline-block;padding:2px 8px;border-radius:9999px;
             margin:2px;background:#2b2b2b;color:#e6e6e6;font-size:12px;}
    </style>
    """
    chips_html = chip_css + "<div>" + "".join(f"<span class='chip'>{t}</span>" for t in present) + "</div>"
    st.markdown(chips_html, unsafe_allow_html=True)

# Cache heavy steps
@st.cache_data(show_spinner=False)
def _cached_read(file):
    return read_uploaded_file(file)

@st.cache_data(show_spinner=False)
def _cached_transform(df: pd.DataFrame, do_basic: bool) -> pd.DataFrame:
    return apply_transformations(df, run_basic_clean=do_basic, label_type_col=None, label_type_vals=None)

# -----------------------------
# App
# -----------------------------
st.set_page_config(page_title="Visualizations Generator", layout="wide")
st.title("Viz Builder")
st.caption("Upload → transform → pick Top-N (per type) → filter by label type → choose brand colors → charts in a grid.")

# -----------------------------
# Sidebar — Upload
# -----------------------------
with st.sidebar:
    st.header("1) Upload file")
    uploaded = st.file_uploader(
        "CSV / XLS / XLSX only",
        type=[e.replace(".", "") for e in sorted(SUPPORTED_EXTS)],
        accept_multiple_files=False,
        help="Other formats will be rejected.",
    )

if uploaded is None:
    st.info("⬅️ Upload a file in the sidebar to begin.")
    st.stop()

# Safe cached read
try:
    df_loaded = _cached_read(uploaded)
except Exception as e:
    with st.sidebar:
        st.error(str(e))
    st.stop()

# Success under the uploader
with st.sidebar:
    st.success(f"Loaded **{uploaded.name}** with **{df_loaded.shape[0]:,}** rows × **{df_loaded.shape[1]:,}** columns.")

# Reserve a slot for chart config (we’ll fill it after df_filtered is computed)
chart_cfg_slot = st.sidebar.container()

# -----------------------------
# Sidebar — Transformations
# -----------------------------
with st.sidebar:
    st.header("2) Transformations")
    do_basic = st.checkbox(
        "Apply basic clean (trim colnames, drop duplicates, trim strings)",
        value=True
    )

# Cached transform
df_transformed = _cached_transform(df_loaded, do_basic=do_basic)

# -----------------------------
# Sidebar — Label Type filter
# -----------------------------
with st.sidebar:
    st.header("3) Label type filter")
    if "labelType" in df_transformed.columns:
        types_all = sorted(df_transformed["labelType"].dropna().astype(str).unique().tolist())
        default_types = ["Noun", "Verb", "Phrase"] if any(t.lower() == "noun" for t in types_all) else types_all
        chosen_types = st.multiselect(
            "Choose label types",
            options=types_all,
            default=default_types,
        )
    else:
        st.info("No `labelType` column found after transformation.")
        chosen_types = []

# -----------------------------
# Sidebar — Top-N per type
# -----------------------------
with st.sidebar:
    st.header("4) Top N labels (global)")
    top_n = st.number_input("How many labels total?", min_value=1, max_value=500, value=25, step=1)
    st.caption("Top is determined by **lift** when available; otherwise by current row order.")

# Consistent 'top' ordering: prefer lift desc when present
df_for_top = df_transformed.copy()
if "lift" in df_for_top.columns and pd.api.types.is_numeric_dtype(df_for_top["lift"]):
    df_for_top = df_for_top.sort_values(by="lift", ascending=False)

# Top-N per selected type
if chosen_types:
    df_filtered = top_labels(df_for_top, types=chosen_types, num=int(top_n))
else:
    df_filtered = df_for_top.copy()

# -----------------------------
# Sidebar — Chart configuration (under Upload)
# -----------------------------
# Defaults from the *filtered* data
numeric_cols = [c for c in df_filtered.columns if pd.api.types.is_numeric_dtype(df_filtered[c])]
non_numeric_cols = [c for c in df_filtered.columns if c not in numeric_cols]

default_label = next(
    (c for c in df_filtered.columns if "labelname" in str(c).lower()),
    next((c for c in df_filtered.columns if str(c).lower() == "label" or "label" in str(c).lower()),
         (non_numeric_cols[0] if non_numeric_cols else df_filtered.columns[0]))
)

default_value = "lift" if ("lift" in df_filtered.columns and pd.api.types.is_numeric_dtype(df_filtered["lift"])) \
                        else (numeric_cols[0] if numeric_cols else df_filtered.columns[0])

with chart_cfg_slot:
    st.header("5) Chart configuration")
    label_col = st.selectbox(
        "Label column",
        options=df_filtered.columns,
        index=list(df_filtered.columns).index(default_label) if default_label in df_filtered.columns else 0,
        key="cfg_label_col",
    )
    value_col = st.selectbox(
        "Value column (numeric)",
        options=df_filtered.columns,
        index=list(df_filtered.columns).index(default_value) if default_value in df_filtered.columns else 0,
        key="cfg_value_col",
    )
    sort_by_value = st.checkbox("Sort bars by value (desc)", value=True, key="cfg_sort_desc")

# -----------------------------
# Sidebar — Brand color theme + preview + WC density
# -----------------------------
with st.sidebar:
    st.header("6) Brand color theme")
    hex_input = st.text_input(
        "Enter HEX color(s)",
        value="#ce1d1d,#fffff0",
        help="One color gradient from light shade to that color. Multiple colors gradient across them."
    )
    # Optional Word Cloud density knobs
    with st.expander("Word Cloud density (optional)", expanded=False):
        wc_max_font = st.slider("Max font (px)", 28, 72, 60, 2)
        wc_padding  = st.slider("Padding (px)", 0, 20, 6, 1)

    # Small inline preview + validation
    hex_error: str | None = None
    try:
        preview_palette = generate_palette(hex_input, n=24)
    except Exception as exc:
        hex_error = str(exc)
        preview_palette = generate_palette("", n=24)

    from PIL import Image, ImageDraw
    box_w, box_h = 20, 20
    width_px = len(preview_palette) * box_w
    img = Image.new("RGB", (width_px, box_h), "white")
    draw = ImageDraw.Draw(img)
    for i, h in enumerate(preview_palette):
        draw.rectangle([i * box_w, 0, (i + 1) * box_w, box_h], fill=h)
    st.image(img, caption=f"Preview ({len(preview_palette)} colors)", width='stretch')

    if hex_error:
        st.error("Invalid HEX color input. Falling back to the default palette.")
        st.caption(hex_error)
    palette_source_hex = hex_input if not hex_error else ""

# -----------------------------
# Build chart data
# -----------------------------
metric = value_col if value_col in df_filtered.columns else None
if metric and pd.api.types.is_numeric_dtype(df_filtered[metric]):
    df_chart = df_filtered.sort_values(
        by=[metric],
        ascending=not sort_by_value,
        kind="mergesort"
    ).reset_index(drop=True)
else:
    df_chart = df_filtered.reset_index(drop=True)

# -----------------------------
# Charts
# -----------------------------
st.markdown("### Charts")
left_opts, right_opts = st.columns([2, 1], vertical_alignment="center")
with left_opts:
    chart_choices = st.multiselect(
        "Choose charts to display",
        ["Circular Bar", "Word Cloud", "Bar Chart"],
        default=["Circular Bar", "Word Cloud" , "Bar Chart"]
    )
with right_opts:
    grid_cols = st.selectbox("Grid columns", options=[1, 2, 3], index=1)

n_rows = int(df_chart.shape[0])
if n_rows == 0:
    st.warning("No rows to display after your filters.")
else:
    palette = generate_palette(palette_source_hex, n=n_rows)
    chart_items: List[Dict[str, Any]] = []


    if "Circular Bar" in chart_choices:
        if not pd.api.types.is_numeric_dtype(df_chart[value_col]):
            st.error(f"Selected value column `{value_col}` is not numeric.")
        else:
            def render_circular(
                data=df_chart,
                label_col_name=label_col,
                lift_col_name=value_col,
                colors=palette,
            ):
                header_left, header_right = st.columns([0.8, 0.2], gap="small")
                with header_left:
                    st.markdown('Hover a bar to see Lift (%).')

                fig_circ = circular_bar_interactive(
                    df=data,
                    label_col=label_col_name,
                    lift_col=lift_col_name,
                    colors=colors,
                    width=900,
                    height=900,
                    label_wrap=20,
                )

                png_bytes = None
                try:
                    png_bytes = generate_circular_bar_chart_png(
                        df=data,
                        label_col=label_col_name,
                        lift_col=lift_col_name,
                        colors=colors,
                        label_wrap=20,
                        width=900,
                        height=900,
                    )
                except ValueError:
                    png_bytes = None
                with header_right:
                    if png_bytes:
                        st.download_button(
                            ' ⬇️ ',
                            data=png_bytes,
                            file_name='circular_bar_chart.png',
                            mime='image/png',
                            width='stretch',
                            key=f"dl_circular_bar_png_{lift_col_name}",
                        )

                st.plotly_chart(fig_circ, use_container_width=True, config={"displayModeBar": False})

            chart_items.append({"title": "Circular Bar", "render": render_circular})


    if "Word Cloud" in chart_choices:
        if not pd.api.types.is_numeric_dtype(df_chart[value_col]):
            st.error(f"Selected value column `{value_col}` is not numeric.")
        else:
            wc_input = df_chart[[label_col, value_col]].rename(
                columns={label_col: "labelName", value_col: "lift"}
            )

            def render_wc(
                data=wc_input,
                colors=palette,
                max_words_wc=min(100, len(wc_input)),
                font_max_wc=int(wc_max_font),
                padding_wc=int(wc_padding),
            ):
                header_left, header_right = st.columns([0.8, 0.2], gap='small')
                with header_left:
                    st.markdown('Hover a word to see Lift (%).')
                result = wordcloud_v2_component(
                    df=data,
                    label_col='labelName',
                    value_col='lift',
                    palette_hexes=colors,
                    width=900,
                    height=500,
                    max_words=max_words_wc,
                    font_min=18,
                    font_max=font_max_wc,
                    padding=padding_wc,
                    prefer_horizontal=1.0,
                    background_color='white',
                    tooltip_label='Lift',
                    component_key=f"wordcloud_main_{value_col}",
                )
                if not result:
                    st.info('Not enough data to render the word cloud.')
                    return
                fallback_fig = result.get('figure')
                image_bytes = result.get('image_bytes')
                with header_right:
                    if image_bytes:
                        st.download_button(
                            ' ⬇️ ',
                            data=image_bytes,
                            file_name='wordcloud.png',
                            mime='image/png',
                            key=f"dl_wordcloud_png_{value_col}",
                            width='stretch',
                        )
                if fallback_fig is not None:
                    st.plotly_chart(fallback_fig, use_container_width=True, config={"displayModeBar": False})

            chart_items.append({"title": "Word Cloud", "render": render_wc})


if "Bar Chart" in chart_choices:
    required = {
        "labelType", "numeratorWhenPresent", "denominatorWhenPresent",
        "numeratorWhenAbsent", "denominatorWhenAbsent"
    }
    missing = [c for c in required if c not in df_transformed.columns]
    if missing:
        st.error(f"Bar chart needs columns missing in data: {missing}")
    else:
        agg_source = df_transformed
        filtered = False
        if chosen_types and 'labelType' in agg_source.columns:
            mask = agg_source['labelType'].astype(str).isin(chosen_types)
            agg_source = agg_source.loc[mask].copy()
            filtered = True
        if agg_source.empty:
            st.info('No data for the selected label types.')
        else:
            agg = aggregate_lift_by_labeltype(agg_source)
            if agg.empty:
                st.info('Aggregation produced no rows for the selected label types.')
            else:
                bar_palette = generate_palette(hex_input, n=max(1, agg.shape[0]))
                title_suffix = 'Selected label types' if filtered else 'Full dataset'

                def render_bar(
                    data=agg,
                    colors=bar_palette,
                    chart_title=f"Lift by Label Type ({title_suffix})",
                ):
                    header_left, header_right = st.columns([0.8, 0.2], gap="small")
                    with header_left:
                        st.markdown('Hover a bar to see Lift (%).')

                    fig_bar = bar_lift_by_type_interactive(
                        df=data,
                        label_col='labelType',
                        lift_col='lift',
                        colors=colors,
                        width=1000,
                        height=600,

                    )

                    png_bytes = None
                    try:
                        png_bytes = generate_bar_lift_by_type_png(
                            df=data,
                            label_col='labelType',
                            lift_col='lift',
                            colors=colors,
                            width=1000,
                            height=600,
                            title=chart_title,
                        )
                    except ValueError:
                        png_bytes = None

                    with header_right:
                        if png_bytes:
                            st.download_button(
                                ' ⬇️ ',
                                data=png_bytes,
                                file_name='bar_lift_by_type.png',
                                mime='image/png',
                                width='stretch',
                                key='dl_bar_lift_png',
                            )

                    st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

                chart_items.append({"title": "Bar Chart Aggregated Lift", "render": render_bar})

    render_fig_grid(chart_items, cols=grid_cols)

# -----------------------------
# Tables (Filtered is default; Transformed tab removed)
# -----------------------------
st.markdown("### Data tables")

# Put FILTERED first so it’s the default visible tab
tab_filtered, tab_loaded = st.tabs(["Filtered (Top-N per type)", "Loaded"])

with tab_filtered:
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Filtered (Top-N per type)")
    if df_filtered.shape[0] == 0:
        st.warning("No rows after Top-N / Label Type filters. Adjust your options in the sidebar.")
    else:
        st.dataframe(df_filtered.head(50), width='stretch')
        st.caption(f"Rows: {df_filtered.shape[0]:,} • Cols: {df_filtered.shape[1]:,}")
        _render_labeltype_chips(df_filtered)

with tab_loaded:
    left, right = st.columns([1, 1])
    st.subheader("Loaded")
    st.dataframe(df_loaded, width='stretch')
    st.caption(f"Rows: {df_loaded.shape[0]:,} • Cols: {df_loaded.shape[1]:,}")
    _render_labeltype_chips(df_loaded)


