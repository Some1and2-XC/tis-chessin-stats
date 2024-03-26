IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='games' and xtype='U')
	CREATE TABLE Games (
        [ID] BIGINT,
		[Event] VARCHAR(255),
		[Site] VARCHAR(255) COLLATE SQL_Latin1_General_CP1_CS_AS PRIMARY KEY,
		[White] VARCHAR(255),
		[Black] VARCHAR(255),
		[Result] VARCHAR(255),
		[UTCDate] DATE,
		[UTCTime] VARCHAR(255),
		[WhiteElo] INTEGER,
		[BlackElo] INTEGER,
		[WhiteRatingDiff] INTEGER,
		[BlackRatingDiff] INTEGER,
		[ECO] VARCHAR(255),
		[Opening] VARCHAR(255),
		[TimeControl] VARCHAR(255),
		[Termination] VARCHAR(255),
		[Moves] TEXT,
		[Date] DATE,
		[Round] VARCHAR(255),
		[WhiteTitle] VARCHAR(255),
		[BlackTitle] VARCHAR(255)
	);

IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='Files' and xtype='U')
	CREATE TABLE Files (
		[filename] VARCHAR(255) PRIMARY KEY
	);
