@ECHO OFF

:: Sets Environment Variables
SET SCHEMA=data\schema.sql
SET DATA_DB=test_db
SET DATA_DB_GAMES=dbo.Games
SET DATA_DB_FILENAMES=dbo.Files

SET DATA_DIR=data
SET BATCH_SIZE=10000
SET DATA_URL=https://database.lichess.org/standard/

:: Setups Database
sqlcmd -d %DATA_DB% -i %SCHEMA%

:: Downloads the Data
ECHO Downloading Data
:: download.py
ECHO Download Finished!

:: Makes the SQL Script
write_sql.py > tmp_file
ECHO Made SQL Script!

:: Adds the data to the DB
sqlcmd -d %DATA_DB% -i tmp_file
del tmp_file

ECHO FINISHED!
PAUSE
EXIT