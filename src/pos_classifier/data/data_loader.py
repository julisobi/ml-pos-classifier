"""Data loader file.

This module provides method for data loading using .csv file.
"""

import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """Load the data and convert the column names.

    Parameters
    ----------
    path : str
        Path to data

    Returns
    -------
    df : pandas.DataFrame
        Loaded data

    """
    df = pd.read_csv(path).rename(columns=lambda x: x.lower().replace(" ", "_"))
    return df
