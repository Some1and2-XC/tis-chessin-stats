#!/usr/bin/bash

# Installs Python Requirements
python3 -m pip install -r requirements.txt

# Installs PGN Extractor
echo Installing PGN-Extract
apt install pgn-extract -y

# Installs Stockfish
echo Installing Stockfish
stockfishurl=https://github.com/official-stockfish/Stockfish/releases/download/sf_16.1/stockfish-ubuntu-x86-64.tar
cd uci
curl -L $stockfishurl > stockfish.tar
tar -xvf stockfish.tar
mv stockfish/stockfish-ubuntu-x86-64 stockfish-ubuntu-x86-64.exe
rm -rf stockfish
cd ..
