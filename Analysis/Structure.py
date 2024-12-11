import pandas as pd

def display_df(df: pd.DataFrame, max_rows: int = 5, max_columns: int = 5) -> None:
    """
    Display the DataFrame with a limited number of rows and columns.

    :param df: The DataFrame to display.
    :param max_rows: The maximum number of rows to display (default value equals to 5).
    :param max_columns: The maximum number of columns to display (default value equals to 5).
    """
    with pd.option_context("display.max_rows", max_rows, "display.max_columns", max_columns):
        print(df)


def visualize_structure(df: pd.DataFrame,name: str) -> None:
    """
    Visualize the structure of the DataFrame.

    :param df: The DataFrame to visualize.
    :param name: The name of the DataFrame.
    """
    print(f"-------------------------------------------\n {name} \n-------------------------------------------")
    df.info()

def nb_unqiue_users(df: pd.DataFrame) -> int:
    """
    Return the number of unique users in the DataFrame.

    :param df: The DataFrame to analyze.
    :return: The number of unique users.
    """
    return df['EmployeeID'].nunique()