#!/usr/bin/env python3

ECO_codes = ["?"]  # Sets a default eco code
for code in "ABCDE":
    for number in range(100):
        ECO_codes.append(f"{code}{number:02d}")
ECO_codes = {
    value: int(index)
    for (index, value) in enumerate(ECO_codes)
}

results = {
    "0-1": 0,
    "1-0": 1,
    "1/2-1/2": 2,
    "*": 3,
}
