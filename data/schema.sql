CREATE TABLE Games (
    [Event] VARCHAR(255),
    [Site] VARCHAR(255),
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
    [Moves] VARCHAR(511),
    [Date] DATE,
    [Round] VARCHAR(255),
    [WhiteTitle] VARCHAR(255),
    [BlackTitle] VARCHAR(255)
);

CREATE TABLE Files (
    [filename] VARCHAR(255) PRIMARY KEY
);