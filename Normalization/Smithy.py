from typing import Literal

import pandas as pd

def merge_dataframe(df1: pd.DataFrame, df2: pd.DataFrame, column: str,method: Literal["left", "right", "inner", "outer", "cross"] = "left") -> pd.DataFrame:
    return pd.merge(df1, df2, on=column, how=method)