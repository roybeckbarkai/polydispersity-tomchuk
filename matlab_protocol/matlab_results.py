from __future__ import annotations

from pathlib import Path

import pandas as pd
from matio import load_from_mat


def load_matlab_results_table(mat_path: str | Path) -> pd.DataFrame:
    table = load_from_mat(str(mat_path), add_table_attrs=True)["Results_Table"].copy()
    if "Winner" in table.columns:
        table["Winner"] = table["Winner"].apply(
            lambda x: x[0] if hasattr(x, "__len__") and not isinstance(x, str) and len(x) else x
        )
    if "V_all" in table.columns:
        table["V_all"] = table["V_all"].apply(lambda x: x.ravel().tolist() if hasattr(x, "ravel") else x)
    return table
