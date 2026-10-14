"""
Stronger training for the Connect 4 network.

Same network as connect4_ann.py, so export_weights.py and the web game keep working.
Improvements over `train` in connect4_ann.py:
  - mirror augmentation: every position is also learned left<->right flipped (2x data, free)
  - deeper search during self-play (depth 3) for better training labels
  - bigger replay buffer and a lower learning rate
  - a gate: every 10 iterations the model plays the previous best; it is only saved if it
    wins at least half. It also reports its score against a random player.

Put this file next to connect4_ann.py and run:
    python train_v2.py                 # 100 iterations x 20 games
    python train_v2.py 300 20          # iterations, games per iteration
"""
import copy
import random
import sys
import numpy as np
import torch
import torch.nn as nn
from connect4_ann import Game, ValueNet, best_move, self_play_game, ROWS, COLS, MODEL_PATH


def mirror(x):
    """Flip an 84-number encoded position left<->right."""
    return x.reshape(2, ROWS, COLS)[:, :, ::-1].reshape(-1).copy()


def play_match(model_a, depth_a, model_b, depth_b, games=20):
    """Score for model_a in [0, games]: win = 1, draw = 0.5. model None = random player."""
    points = 0.0
    for g in range(games):
        game = Game()
        a_is_p1 = g % 2 == 0
        for _ in range(2):                                   # random opening for variety
            game.play(random.choice(game.legal_moves()))
        while not game.is_over():
            a_turn = (game.player == 1) == a_is_p1
            model, depth = (model_a, depth_a) if a_turn else (model_b, depth_b)
            if model is None:
                col = random.choice(game.legal_moves())
            else:
                col = best_move(model, game, depth)
            game.play(col)
        if game.winner == 0:
            points += 0.5
        elif (game.winner == 1) == a_is_p1:
            points += 1
    return points


def train(iterations=100, games_per_iter=20, epochs=3, batch_size=256, buffer_max=60000, depth=3):
    model = ValueNet()
    try:
        model.load_state_dict(torch.load(MODEL_PATH))
        print("Continuing from saved model.")
    except FileNotFoundError:
        pass
    model.eval()
    best = copy.deepcopy(model).eval()
    opt = torch.optim.Adam(model.parameters(), lr=5e-4)
    loss_fn = nn.MSELoss()
    buf_x, buf_y = [], []

    for it in range(1, iterations + 1):
        model.eval()
        for _ in range(games_per_iter):
            X, y, _ = self_play_game(model, depth=depth, epsilon=0.2)
            buf_x += X + [mirror(x) for x in X]
            buf_y += y + y
        buf_x, buf_y = buf_x[-buffer_max:], buf_y[-buffer_max:]

        Xt = torch.from_numpy(np.array(buf_x, dtype=np.float32))
        yt = torch.tensor(buf_y, dtype=torch.float32).unsqueeze(1)
        model.train()
        for _ in range(epochs):
            perm = torch.randperm(len(Xt))
            for i in range(0, len(Xt), batch_size):
                idx = perm[i:i + batch_size]
                opt.zero_grad()
                loss = loss_fn(model(Xt[idx]), yt[idx])
                loss.backward()
                opt.step()
        model.eval()
        print(f"iter {it}/{iterations}  buffer={len(buf_x)}  loss={loss.item():.4f}")

        if it % 10 == 0:
            vs_random = play_match(model, 2, None, 0) / 20
            vs_best = play_match(model, 2, best, 2) / 20
            print(f"   check: vs random {vs_random:.0%} | vs previous best {vs_best:.0%}")
            if vs_best >= 0.5:
                best = copy.deepcopy(model).eval()
                torch.save(best.state_dict(), MODEL_PATH)
                print("   -> new best model saved")
    print("Done. Best model is in", MODEL_PATH)


if __name__ == "__main__":
    train(*[int(a) for a in sys.argv[1:3]])
