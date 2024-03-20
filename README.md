# T'is chessin stats
Stats project for chess (now with AI!)
## Least Played ECO Codes:
 - C76
 - D65
 - A98
 - A78
 - A79

## Duplicate Game Codes:
```
SELECT TOP(15) COUNT(Site), Site FROM Games
GROUP BY Site
ORDER BY COUNT(Site) DESC
;
```
 - https://lichess.org/RTY1cld5
 - https://lichess.org/rTy1cLD5

 - https://lichess.org/ZUgRtCFE
 - https://lichess.org/ZugRtCfe
## Collation for case sensitivity
For SQL Server, you can set the character set to use on data.
Because of this you can make some columns case sensitive and some columns case insensitive. 
You can also decide if a column is going to be case sensitive or insensitive in a query. 
Some examples of character sets: `SQL_Latin1_General_CP1_CS_AS` & `SQL_Latin1_General_CP1_CI_AS`.
The former is case sensitive (the CS in its name,) the latter is case insensitive (the CI in its name.)
### Case Sensitive Duplicates
The following query uses the COLLATE keyword to find duplicate games (the first 1000.)
```
SELECT * FROM (
	SELECT TOP(1000)
		COUNT(Games_CS.Site) [count],
		Games_CS.Site [site]
	FROM
		(SELECT
			(Site COLLATE SQL_Latin1_General_CP1_CS_AS) Site
		FROM Games) Games_CS
	GROUP BY Games_CS.Site
	ORDER BY COUNT(Games_CS.Site) DESC) data
WHERE data.count != 1
```
## What the files are for
### automation.bat
This is the main controller script for the entire program. This will run everything from downloading the data to running the `ai.py` file.
This also sets environment variables to help manage things like which directory should the data go do and which database etc.
### download.py
This file does what it sounds like, it downloads the chess data. The way it does this however is clever. 
This script will take streams of data instead of entire files and decompresses the files into CSVs that include only sections of the full dataset.
It does this several times (per available dataset file from [lichess](https://database.lichess.org/)) and fills the `data/` directory with these CSVs.
### ai.py
This file is responsible for all things AI. This includes the following:
 - Setting up the AI model to be able to make predictions.
 - Setting up the dataset for training the model.
 - Adding extra parameters to help the AI along (things like getting a position and taking the eval and move number.)
### write_sql.py
This file is designed to work with MS SQL Server. This script prints out a SQL script to be able to add all the data from each CSV file from the data directory into the database. 
### casting.py
All this file does is have a list of the different columns the main output data would have (not derived columns such as eval.)