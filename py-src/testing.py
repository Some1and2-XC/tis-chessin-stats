#!/usr/bin/env python3

# from tis_chessin_stats import test_func  # This import is for if the rust version if working

import io
import json

from sys import argv
from glob import glob
from Lindex import lindex
import zstandard as zstd


def get_filenames() -> str:
    """
    Sets the filename to load the file from
    """

    if len(argv) >= 2:
        files = argv[1:]
    else:
        print(f"No Arguments Passed:")
        print(" - (You need to set a filename as a CLI argument!)")
        print(f" - Defaulting to `{FILE_EXTENSION}` files found in directory!")

        files = glob("*" + FILE_EXTENSION)

        if len(files) >= 1:
            for file in files:
                print(f"    - Found `{file}`")
            print()

        else:  # Exits the program
            print(f" - No file ending in `{FILE_EXTENSION}` FOUND! (exiting)")
            exit()  # Ends script execution
    
    return files


def load_file(filename: str, method):
    """
    Loads the file in (only sections of the file at a time)
    """

    try:
        with open(filename, "rb") as fh:  # Loads file in
            dctx = zstd.ZstdDecompressor()
            stream_reader = dctx.stream_reader(fh)
            text_stream = io.TextIOWrapper(stream_reader, encoding="utf-8")

            local_list = []
            for line in text_stream:
                # if line == "":
                #     input(local_list)
                #     local_list = []

                method(line)

    except Exception as e:
        raise e


def handle_lines(line: str):
    """
    Defines the method for what to do to each line being passed in
    """

    global COUNT  # Uses a global line counter
    global FILE_SPECIFIED

    print(f" - File: {FILE_SPECIFIED} ~ Line: {COUNT}", end="\r")
    COUNT += 1

    # print(line.strip(), end="\r")
    # obj = json.loads(line)
    # lindex(obj).pprint()


FILE_EXTENSION = ".zst"
FILE_SPECIFIED = ""
COUNT = 0


if __name__ == "__main__":

    files = get_filenames()
    method = handle_lines
    for i, filename in enumerate(files):
        FILE_SPECIFIED = filename
        load_file(filename=filename, method=method)
        print()