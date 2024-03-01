"""
File for casting to different outputs
"""

ATTRIBUTES = [
    "Event",
    "Site",
    "White",
    "Black",
    "Result",
    "UTCDate",
    "UTCTime",
    "WhiteElo",
    "BlackElo",
    "WhiteRatingDiff",
    "BlackRatingDiff",
    "ECO",
    "Opening",
    "TimeControl",
    "Termination",
]


class Entry:
    def __init__(self, attributes):
        for k, v in attributes.items():
            if k in ATTRIBUTES:
                setattr(self, k, v)

class Output:
    def __init__(self, method):
        self.get_output = method


def get_csv_value(entry, first = False):
    if first: yield ",".join(ATTRIBUTES) + "\n"
    else:
        yield ",".join(
            str(
                getattr(
                    entry,
                    k
                )
            )
            for k in ATTRIBUTES) \
            + "\n"


CSV = Output(method = get_csv_value)