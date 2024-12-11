import os
import pandas as pd

def __exist_file__(path):
    """
    Check if a file exists
    :param path: Path to the file
    :return: True if the file exists, False otherwise
    """
    return os.path.exists(path)

def __is_csv__(path):
    """
    Check if a file is a csv file
    :param path: Path to the file
    :return: True if the file is a csv file, False otherwise
    """
    return path.endswith('.csv')

def create_dataframe(path: str) -> pd.DataFrame:
    """
    Read a csv file
    :param path: Path to the csv file
    :return: DataFrame with the csv data
    :raises FileNotFoundError: If the file does not exist
    """
    if __exist_file__(path):
        if not __is_csv__(path): raise ValueError(f"File {path} is not a csv file")
        return pd.read_csv(path)
    else: raise FileNotFoundError(f"File {path} not found")