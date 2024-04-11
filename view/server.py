#!/usr/bin/env python3

from flask import Flask, render_template, redirect, request

import xgboost as xgb
import numpy as np

import os
import io

import data

import chess.engine
import chess.pgn


os.environ["CHESS_ENGINE"] = "stockfish-windows-x86-64-avx2.exe"

app = Flask(__name__)


def get_eco_code(pgn: str) -> str:
    """
    Function for getting the ECO code of the game.
    """
    import subprocess
    import json

    parser = ["../../pgn-extract.exe", "-e", "--json"]
    proc = subprocess.Popen(args=parser, stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    pgn += " 1-0"
    out = proc.communicate(input=bytearray(pgn, "ascii"))

    print(out, type(out))

    if out[0] is None:
        raise ValueError("Can't Parse Game")
        return

    return (json.loads(out[0])[0]["ECO"], json.loads(out[0])[0]["Opening"])


@app.route("/", methods=("GET",))
def home():
    return render_template("index.html")


@app.route("/<filename>.js", methods=("GET",))
def js_files(filename):
    return render_template(filename + ".js")


@app.route("/img/chesspieces/wikipedia/<filename>.png")
def image(filename):
    return redirect(f"https://chessboardjs.com/img/chesspieces/wikipedia/{filename}.png")


@app.route("/get_stats/")
def stats():

    # Initializes Chess Data
    WhiteElo = request.args.get("whiteElo", default=1500, type=int)
    BlackElo = request.args.get("BlackElo", default=1500, type=int)

    try:
        eco, opening = get_eco_code(request.args.get("pgn", type=str))
    except:
        eco, opening = "?", "?"

    engine: chess.engine.SimpleEngine = chess.engine.SimpleEngine.popen_uci(os.path.join("..", "uci", os.environ["CHESS_ENGINE"]))
    game: chess.pgn.Game = chess.pgn.read_game(io.StringIO(request.args.get("pgn", type=str)))
    board: chess.Board = chess.Board()

    try:
        moves = [move for move in game.mainline_moves()]
        amnt_of_moves = len(moves)

        for move in moves:
            board.push(move)
    except:
        amnt_of_moves = 0

    eval = engine \
        .analyse(board, chess.engine.Limit(depth=16))["score"] \
        .white() \
        .score(mate_score=100000)

    ECO_code_arr = [0 for i in data.ECO_codes]
    ECO_code_arr[data.ECO_codes[eco]] = 1

    AI_arr = [WhiteElo, BlackElo, amnt_of_moves, eval, *ECO_code_arr]

    model = xgb.XGBRegressor()
    model.load_model("../new_output/model.json")

    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    prediction = softmax(model.predict([AI_arr])[0][:3])

    results = {
        "0-1": 0,
        "1-0": 1,
        "1/2-1/2": 2,
    }

    prediction = {
        k: float(prediction[v])
        for k, v in results.items()
    }

    return {
        "eval": eval,
        "prediction": prediction,
        "eco": eco,
        "opening": opening,
    }


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)