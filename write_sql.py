#!/usr/bin/env python3

"""
Python file for making a SQL file for importing data
"""

from glob import glob
from os import path

directory = "data"  # The folder that holds the CSV files
data_file_meta = "dbo.Files"  # The name of the table the meta data for which files were added
table_name = "dbo.Games"  # The database table to put the data into

files = glob(path.join(directory, "*.csv"))  # Gets a list of directories for the CSVs

out_txt = [
f"""
IF NOT EXISTS (
    SELECT * FROM {data_file_meta}
    WHERE filename='{file}'
)
BEGIN
    INSERT INTO {data_file_meta} (filename) VALUES ('{file}');

    BULK INSERT {table_name}
    FROM '{path.abspath(file)}'
    WITH (FORMAT='CSV', FIRSTROW=2);
END;

GO
"""
    for file in files
]

print("".join(out_txt))
