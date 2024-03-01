#!/usr/bin/env python3

# from tis_chessin_stats import test_func  # This import is for if the rust version if working

import io
import json

import lxml.html
import requests

from sys import argv
from glob import glob
from re import findall as re
from Lindex import lindex
import zstandard as zstd

import casting


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

    count = 0
    move_index = "Moves"  # The dict index of the moves of the game
    show_lines = 1

    fh = requests.get(url, stream=True).raw

    dctx = zstd.ZstdDecompressor()
    stream_reader = dctx.stream_reader(fh)
    text_stream = io.TextIOWrapper(stream_reader, encoding="utf-8")

    output = dict()
    for line in text_stream:
        if show_lines:
            handle_lines(line, count)
            count += 1

            continue

        line = line.strip()

        if line == "" and move_index in output:
            lindex(output).pprint()
            input()
            output = dict()
            continue

        # Parses pgn files
        if line.startswith("[") and line.endswith("\"]"):
            [k, v] = line[1:-2].split(" \"")
            output[k] = v
            continue

        output[move_index] = line
    handle_lines(line, count, last=True)


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

DATA_URL = "https://database.lichess.org/standard/"


if __name__ == "__main__":

    urls = get_network_filenames()
    AMNT_OF_FILES = len(urls)

    # files = get_filenames()

    method = handle_lines
    for i, url in enumerate(urls):
        FILE_SPECIFIED = url
        CURRENT_FILE_IDX = i

        load_file(url=url, method=method)
        print()
        # print()
        # exit()