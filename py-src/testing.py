#!/usr/bin/env python3

# from tis_chessin_stats import test_func  # This import is for if the rust version if working

import io
import json
from sys import argv
from glob import glob
from Lindex import lindex
import zstandard as zstd


FILE_EXTENSION = ".zst"

if __name__ == "__main__":


    # Sets the filename to load the file from
    try:
        filename = argv[1]
    except IndexError as e:
        print(f"IndexError: {e}")
        print(" - (You need to set a filename as a CLI argument!)")
        print(f" - Defaulting to first `{FILE_EXTENSION}` file found in directory!")

        try:
            filename = glob("*" + FILE_EXTENSION)[0]
            print(f" - Found `{filename}`")
        except IndexError as e:
            print(f" - No file ending in `{FILE_EXTENSION}` FOUND! (exiting)")
            exit()  # Ends script execution


    # Loads the file in (only sections of the file at a time)
    try:
        with open(filename, "rb") as fh:
            dctx = zstd.ZstdDecompressor()
            stream_reader = dctx.stream_reader(fh)
            text_stream = io.TextIOWrapper(stream_reader, encoding="utf-8")
            for line in text_stream:
                print(line)
                # obj = json.loads(line)
                # lindex(obj).pprint()
                input()

    # Handles any exception
    except Exception as e:
        raise e
        # print(f"Error: {e}")