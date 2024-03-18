#!/bin/bash

# Sets Environment Variables
set -a

DATA_DIR = "data"
BATCH_SIZE = 10000
DATA_URL = "https://database.lichess.org/standard/"

set +a

./download.py
sql_script="$(./write_sql.py)'

sqlcmd -d lichess -Q "${sql_script}"