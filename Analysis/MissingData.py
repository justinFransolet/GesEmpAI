import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt

def textual_missing_data(df: pd.DataFrame, name: str) -> None:
    """
    Display the missing data in a textual format.

    :param df: The DataFrame to analyze.
    :param name: The name of the DataFrame.
    """
    # Format the missing data
    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0]
    missing_data = missing_data.sort_values(ascending=False)
    # Display the missing data
    print(f"Valeurs manquantes dans {name} :\n----------------------------")
    print(f"{missing_data}\n")

def visual_missing_data(df: pd.DataFrame, name: str) -> None:
    """
    Display the missing data in a visual format.

    :param df: The DataFrame to analyze.
    :param name: The name of the DataFrame.
    """
    print(f"Valeurs manquantes dans {name} :\n----------------------------")
    msno.matrix(df)
    plt.show()