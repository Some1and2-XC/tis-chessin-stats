#!/usr/bin/env python3

import pandas as pd
import os

from pgn_parser import pgn, parser
from casting import ATTRIBUTES

from glob import glob


def load_data() -> pd.DataFrame:
    """
    Function for loading in the data from the data folder
    """

    data_folder = os.environ["DATA_DIR"]

    df = None

    for file in glob(os.path.join(data_folder, "*.csv")):

        if type(df) == None:
            df = pd.read_csv(file)
        else:
            df = pd.concat([df, pd.read_csv(file)], ignore_index=True)

    print(df)

    return df


def parse_moves() -> pd.DataFrame:
    """
    Function for parsing the moves from a dataframe
    """


def make_model():
    """
    Function for creating an AI model
    """


if __name__ == "__main__":
    os.environ["DATA_DIR"] = "data"
    data = load_data()
