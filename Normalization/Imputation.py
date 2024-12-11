import pandas as pd

def impute_median(df: pd.DataFrame, column: str) -> None:
    """
    Impute the median value for missing values in the specified columns

    :param df: DataFrame to impute missing values
    :param column: Column to impute missing values
    """

    median_num_comp = df[column].median()
    df[column].fillna(median_num_comp, inplace=True)