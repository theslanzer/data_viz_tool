from __future__ import annotations
import os
import pandas as pd

SUPPORTED_EXTS = {".csv", ".xls", ".xlsx"}

def _ext(filename: str) -> str:
    return os.path.splitext(filename.lower())[1]

def read_uploaded_file(uploaded_file) -> pd.DataFrame:
    if uploaded_file is None:
        raise ValueError("No file uploaded.")

    ext = _ext(uploaded_file.name)
    if ext not in SUPPORTED_EXTS:
        raise ValueError(
            f"File format not supported: {ext}. "
            f"Please upload one of: {', '.join(sorted(SUPPORTED_EXTS))}"
        )

    if ext == ".csv":
        try:
            return pd.read_csv(uploaded_file)
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, encoding="latin-1")
    else:
        return pd.read_excel(uploaded_file)
