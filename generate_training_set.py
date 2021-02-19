import os
import chess.pgn
import numpy as np
from state import State

def get_dataset(num_samples=None):
    X,Y = [], []
    game_number = 0
    for fn in os.listdir("data"):
        pgn = open(os.path.join("data",fn))
        
        while 1:
            try:
                game = chess.pgn.read_game(pgn)
            except Exception:
                break
            result = {'1/2-1/2':0, '0-1':-1, '1-0':1}[game.headers['Result']]
            board = game.board()
            for i, move in enumerate(game.mainline_moves()):
                board.push(move)
                serialized = State(board).serialize()
                X.append(serialized)
                Y.append(result)
            print("Parsing training game %d, got %d examples" % (game_number, len(X)))
            if num_samples is not None and len(X) > num_samples:
                return X, Y
            game_number += 1
    X = np.array(X)
    Y = np.array(Y)
    print(X, Y)
    return X,Y

if __name__ == "__main__":
    X, Y = get_dataset(5000000)
    np.savez("processed/dataset_5M.npz", X, Y)
    