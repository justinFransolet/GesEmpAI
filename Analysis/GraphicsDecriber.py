import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import missingno as msno

def display_graphics_numeral_column(df: pd.DataFrame, column: str) -> None:
    """
    Display graphics for a given column of a dataframe

    :param df: The dataframe to analyze
    :param column: The name of the numeral column to analyze
    """
    plt.figure(figsize=(4,2))
    sns.histplot(data=df, x=column, kde=True)
    plt.title(f"Distribution de {column}")
    plt.show()

def display_graphics_cat_column(df: pd.DataFrame, column: str) -> None:
    """
    Display graphics for a given column of a dataframe

    :param df: The dataframe to analyze
    :param column: The name of the categorical column to analyze
    """
    plt.figure(figsize=(4, 2))
    df[column].value_counts().plot(kind='bar')
    plt.title(f"Répartition de {column}")
    plt.xlabel(column)
    plt.ylabel("Count")
    plt.show()

def display_heatmap_numeric_values(df: pd.DataFrame, columns: [str]) -> None:
    """
    Display a heatmap of the correlation between numeric columns

    :param df: The dataframe to analyze
    :param columns: The list of numeric columns to analyze
    """
    plt.figure(figsize=(5,4))
    msno.heatmap(df[columns])
    plt.title("Heatmap des corrélations entre variables numériques")
    plt.show()

def display_box_plot(df: pd.DataFrame, x: str, y: str) -> None:
    """
    Display a box plot for a given dataframe and two columns

    :param df: The dataframe to analyze
    :param x: The name of the x column
    :param y: The name of the y column
    """
    plt.figure(figsize=(4,2))
    sns.boxplot(x=x, y=y, data=df)
    plt.title(f"Distribution de {y} selon {x}")
    plt.show()

def display_graphics_contingency(df: pd.DataFrame, column1: str, column2: str) -> None:
    """
    Display graphics for a given contingency table

    :param df: The dataframe to analyze
    :param column1: The name of the first column
    :param column2: The name of the second column
    """
    contingency = pd.crosstab(df[column1], df[column2])
    print(f"Table de contingence\n{contingency}")
    contingency.plot(kind='bar', stacked=True, figsize=(6,4))
    plt.title(f"{column1} vs {column2}")
    plt.xlabel(column1)
    plt.ylabel("Count")
    plt.show()