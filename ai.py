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


def make_model(df: pd.DataFrame, test_size: float = 0.1):
    """
    Function for creating an AI model

    df is the dataframe with all the chess data inside
    test_size is the portion of the dataset that becomes test data
    """

    def get_one_hot(col, values: dict) -> pd.DataFrame:

        """
        Gets one hot encoded np array for values
        the df should be remaped with values before calling this function
        """

        # Initializes the array
        arr = np.zeros((len(col), len(values)))

        # Sets the associated value for each item
        arr[np.arange(len(col)), col] = 1

        df = pd.DataFrame(arr, columns=values.keys())

        return df

    import sklearn.model_selection as sk
    import numpy as np

    # One-hot encodes results
    results = {
        "0-1": 0,
        "1-0": 1,
        "1/2-1/2": 2,
        "*": 3,
    }
    df["Result"] = df["Result"].map(results)
    y = get_one_hot(df["Result"], results)

    # initializes the df with less attributes
    df = df[["WhiteElo", "BlackElo", "ECO", "MoveN", "Eval"]]

    # One-hot encodes eco codes
    ECO_codes = ["?"]  # Sets a default eco code

    for code in "ABCDE":
        for number in range(100):
            ECO_codes.append(f"{code}{number:02d}")
    
    ECO_codes = {
        value: int(index)
        for (index, value) in enumerate(ECO_codes)
    }
    df["ECO"] = df["ECO"].map(ECO_codes)
    eco_codes = get_one_hot(df["ECO"], ECO_codes)
    df = pd.concat([df, eco_codes], axis=1)

    # Removes the unwanted columns
    df.drop(columns=["ECO"])  # Drops original eco code column and result column

    """
    # K Fold Validation
    # Using sk libs for training means I can't use internal model.save_model() method
    print(f"Accuracy : {np.mean(n_scores):.3f}% ({np.std(n_scores):.3f} std)")
    """

    # https://mljar.com/blog/xgboost-save-load-python/
    # Splits the data into test & training
    x_train, x_test, y_train, y_test = sk.train_test_split(df, y, test_size=test_size)
    train = xgb.DMatrix(x_train, y_train)
    test = xgb.DMatrix(x_test, y_test)

    # Initializes the model
    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.01,
        early_stopping_rounds=20,
    )

    # Tries to fit the data
    model.fit(x_train, y_train,
        eval_set=[(x_train, y_train), (x_test, y_test)]
    )

    res = model.predict(x_test, iteration_range=(0, model.best_iteration + 1))
    # print(res)

    print("Best Iteration:", model.best_iteration)


    return model


CHESS_AI_DEPTH = 16  # Depth ~16 takes ~0.1s
DEVELOPMENT = False
OUTPUT = "output"


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

        with open(os.path.join(OUTPUT, filename), "wb") as handle:
            pickle.dump(data, handle)
        print(f" - Dataset Saved! ('{filename}')")
    
    with open(os.path.join(OUTPUT, "dataset.pkl"), "rb") as f:
        data = pickle.loads(f.read())

    model = make_model(data)

    # model.save_model(os.path.join(OUTPUT, "model.json"))

    print("Finished!")
