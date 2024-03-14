#!/usr/bin/env python3

# from tis_chessin_stats import test_func  # This import is for if the rust version if working

import pandas as pd

import io
import os
import json

import lxml.html
import requests

from sys import argv
from glob import glob
from re import findall as re

from Lindex import lindex
import zstandard as zstd

from casting import ATTRIBUTES


def get_network_filenames() -> list[str]:
    """
    Gets URLs which point to resources to download
    """

    global DATA_URL, FILE_EXTENSION

    res = requests.get(DATA_URL)
    html = lxml.html.fromstring(res.text)  # Loads lxml.html object
    links = html.xpath("//a/@href")  # Gets all the links

    # Gets links that have the correct extension & adds the URL to them
    links = [
        DATA_URL + link \
        for link in links \
            if link.endswith(FILE_EXTENSION) ]
    
    if 0: [print(" -", link) for link in links]

    return links


def load_file(url: str, method):
    """
    Loads the file in (only sections of the file at a time)
    """

    def get_df_from_dict(data: dict) -> pd.DataFrame:
        """
        Converts from a dictionary to pandas dataframe
        """

        return pd.DataFrame({i: [data[i]] for i in data})


    count = 0
    full_count = 0
    move_index = "Moves"  # The dict index of the moves of the game
    show_lines = 0

    fh = requests.get(url, stream=True).raw

    dctx = zstd.ZstdDecompressor()
    stream_reader = dctx.stream_reader(fh)
    text_stream = io.TextIOWrapper(stream_reader, encoding="utf-8")

    output = dict()

    df = pd.DataFrame({k: [] for k in ATTRIBUTES})

    # Goes through each line from the downloaded stream
    for line in text_stream:


        line = line.strip()

        if line == "" and move_index in output:

            df = pd.concat([df, get_df_from_dict(output)], ignore_index=True)

            count += 1
            full_count += 1

            if count >= ENTRIES_PER_OUTPUT:
                filename = f"{url.lstrip(DATA_URL)}_#{full_count}.csv"
                df.to_csv(
                    os.path.join(OUT_FOLDER, filename),
                    encoding="utf-8")
                print(f" - Wrote file: '{filename}'", end="\r")
                df = pd.DataFrame({k: [] for k in ATTRIBUTES})  # Clears dataframe
                count = 0

            output = dict()

            continue

        # Parses pgn files
        if line.startswith("[") and line.endswith("\"]"):
            [k, v] = line[1:-2].split(" \"")
            output[k] = v
            continue

        output[move_index] = line

    df.to_csv(f"{url.lstrip(DATA_URL)}_#{full_count}.csv", encoding="utf-8")


def handle_lines(line: str, count, last = False):
    """
    Defines the method for what to do to each line being passed in
    """

    global FILE_SPECIFIED, AMNT_OF_FILES, CURRENT_FILE_IDX

    if count % 1000 == 0 or last:
        print(f" - [File: {CURRENT_FILE_IDX + 1} / {AMNT_OF_FILES}] Filename: {FILE_SPECIFIED} ~ Line: {count}", end="\r")


FILE_EXTENSION = ".zst"
FILE_SPECIFIED = ""  # Current Filename

AMNT_OF_FILES = 0  # Total Amount of Files
CURRENT_FILE_IDX = 0  # Current File Index

ENTRIES_PER_OUTPUT = 10000

DATA_URL = "https://database.lichess.org/standard/"
OUT_FOLDER = "data"


if __name__ == "__main__":

    urls = get_network_filenames()[::-1]
    AMNT_OF_FILES = len(urls)

    method = handle_lines
    for i, url in enumerate(urls):
        FILE_SPECIFIED = url
        CURRENT_FILE_IDX = i

        load_file(url=url, method=method)
        print()
        break
