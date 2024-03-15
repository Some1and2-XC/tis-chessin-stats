#!/usr/bin/env python3

"""
Python file for making a SQL file for importing data
"""

from glob import glob
from os import path

directory = "data"  # The folder that holds the CSV files
table_name = "dbo.Games"  # The database table to put the data into

files = glob(path.join(directory, "*.csv"))  # Gets a list of directories for the CSVs

out_txt = [
f"""
BULK INSERT {table_name}
FROM '{path.abspath(file)}'
WITH (FORMAT='CSV', FIRSTROW=2)

GO
"""
    for file in files
]

print("".join(out_txt))
