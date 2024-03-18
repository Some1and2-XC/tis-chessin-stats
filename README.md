# T'is chessin stats
Stats project for chess (now with AI!)
## LEAST PLAYED ECO CODES:
 - C76
 - D65
 - A98
 - A78
 - A79

## DUPLICATE GAME CODES:
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
### CASE SENSITIVE DUPLICATES
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
