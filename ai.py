#!/usr/bin/env python3

import pandas as pd
import xgboost as xgb

import os
import io
import random
import pickle
import time

import chess.engine
import chess.pgn

from casting import ATTRIBUTES

from glob import glob


def load_data() -> pd.DataFrame:
    """
    Function for loading in the data from the data folder
    """

    # Initializes DataFrame`
    data_folder = os.environ["DATA_DIR"]
    df = None
    for file in glob(os.path.join(data_folder, "*.csv")):

        if type(df) == None:
            df = pd.read_csv(file)
        else:
            df = pd.concat([df, pd.read_csv(file).head(200)], ignore_index=True)

    df = parse_moves(df)

    return df


def parse_moves(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function for parsing the moves from a dataframe
    """


    def get_move_chosen(col) -> int:
        game: chess.pgn.Game = chess.pgn.read_game(io.StringIO(col.Moves))
        game_length = len(list(game.mainline_moves()))

        if game_length == 0:
            return None
        else:
            return random.randint(0, game_length - 1)


    def get_eval(col) -> int:
        game: chess.pgn.Game = chess.pgn.read_game(io.StringIO(col.Moves))
        board: chess.Board = chess.Board()

        # loads the board in
        for i, move in enumerate(game.mainline_moves()):
            board.push(move)
            if i == col.MoveN:
                break

        info = engine.analyse(board, chess.engine.Limit(depth=CHESS_AI_DEPTH))

        return info["score"].white().score(mate_score=100000)


    if DEVELOPMENT:
        df = df.head(10)

    df = df.assign(MoveN=df.apply(get_move_chosen, axis=1))
    print(" - Added Move Number!")
    df = df.assign(Eval=df.apply(get_eval, axis=1))
    print(" - Added Eval Values!")

    return df


def make_model(df: pd.DataFrame):
    """
    Function for creating an AI model
    """
    
    ECO_codes = ["?"]

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

    df["ECO"] = df["ECO"].map(ECO_codes)
    df["Result"] = df["Result"].map(results)
    df = df[["WhiteElo", "BlackElo", "ECO", "MoveN", "Eval", "Result"]]

    print(df)


CHESS_AI_DEPTH = 9  # Depth ~16 takes ~0.1s
DEVELOPMENT = False


if __name__ == "__main__":
    os.environ["DATA_DIR"] = "data"

    if DEVELOPMENT:

        engine: chess.engine.SimpleEngine = chess.engine.SimpleEngine.popen_uci("./uci/stockfish-windows-x86-64-avx2.exe")

        print("Loading Data... (This might take a while)")
        start = time.perf_counter()
        data = load_data()
        print(f"Time Elapsed: {time.perf_counter() - start:.5f}s")

        print("Saving Files!")
        engine.quit()
        print(" - Engine Saved!")

        if DEVELOPMENT:
            filename = "dataset_dev.pkl"
        else:
            filename = "dataset.pkl"

        with open(filename, "wb") as handle:
            pickle.dump(data, handle)
        print(f" - Dataset Saved! ('{filename}')")
    
    with open("dataset_dev.pkl", "rb") as f:
        data = pickle.loads(f.read())

    make_model(data)

    print("Finished!")
