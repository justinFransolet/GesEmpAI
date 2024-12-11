import pandas as pd

def display_describe(df: pd.DataFrame) -> None:
    """
    Display the describe of the dataframe

    :param df: The dataframe to describe
    """
    print(df.describe())

def describe_values_columns(df: pd.DataFrame, columns: str) -> None:
    """
    Display the different values statistics of the columns

    :param df: The dataframe to describe the columns
    :param columns: The name of the columns to describe values
    """
    print(df[columns].value_counts())

def describe_columns(df: pd.DataFrame, columns: str) -> None:
    """
    Display the different statistics of the columns

    :param df: The dataframe to describe the columns
    :param columns: The name of the columns to describe
    """
    print(df[columns].describe())