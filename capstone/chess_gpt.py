"""
Capstone starter: train a GPT from scratch to play (legal) chess.
===============================================================

Same architecture as [2.3] Shakespeare from Scratch -- only the *data* and the
*metric* change. The point of the project is that you get an objective score,
not a vibe check: after each checkpoint we ask python-chess how many moves the
model can produce before it plays something illegal.

    python3 capstone/chess_gpt.py                 # ~2 min on a laptop CPU
    python3 capstone/chess_gpt.py --steps 3000    # better results
    ezpz launch python3 capstone/chess_gpt.py     # multi-GPU

THE LESSON (already measured -- reproduce it, do not take our word for it):

    char-level tokens : legal-move prefix stays ~1-2 no matter how low the loss goes
    move-level tokens : legal-move prefix climbs 10 -> 17 over the same budget

A character model learns what SAN *looks like* ("Nf3", "Bxc4+") but has no idea
where the pieces are, so it emits well-formed illegal moves forever. Giving it
one token per move lets it model the game instead of the spelling. Run both
(`--tokens char` vs `--tokens move`) and put the two curves on one slide.

Deps:  pip install torch python-chess
"""

from __future__ import annotations

import argparse
import io
import os
import pickle
import time
import urllib.request

import chess
import chess.pgn
import torch
import torch.nn as nn
import torch.nn.functional as F

# A 1.4 MB PGN of real tournament games (TWIC issue 1000).
PGN_URL = (
    "https://raw.githubusercontent.com/rozim/ChessData/master/Twic/twic1000.pgn"
)
CACHE = "/tmp/chess_games.pkl"


# --------------------------------------------------------------------------
# 1. Data: PGN -> one line of SAN moves per game
# --------------------------------------------------------------------------
def load_games(n_games: int = 1500) -> list[str]:
    """Download the PGN once, parse it into 'e4 c5 Nf3 d6 ...' strings."""
    key = f"{CACHE}.{n_games}"
    if os.path.exists(key):
        return pickle.load(open(key, "rb"))

    print(f"downloading {PGN_URL} ...")
    raw = urllib.request.urlopen(PGN_URL, timeout=60).read().decode(
        "utf-8", errors="ignore"
    )

    fh, games = io.StringIO(raw), []
    while len(games) < n_games:
        g = chess.pgn.read_game(fh)
        if g is None:
            break
        board, sans = g.board(), []
        for mv in g.mainline_moves():
            sans.append(board.san(mv))  # SAN must be read BEFORE the push
            board.push(mv)
        if len(sans) >= 10:  # skip abandoned games
            games.append(" ".join(sans))

    pickle.dump(games, open(key, "wb"))
    return games


def build_dataset(games: list[str], mode: str):
    """Return (flat_token_tensor, vocab, stoi).

    mode="char": vocab is ~29 characters.  The model must spell each move.
    mode="move": vocab is ~2000 whole moves. The model predicts a move at a time.
    """
    if mode == "char":
        text = "\n".join(games)
        vocab = sorted(set(text))
        stoi = {c: i for i, c in enumerate(vocab)}
        flat = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    else:
        vocab = sorted({m for g in games for m in g.split()})
        stoi = {m: i for i, m in enumerate(vocab)}
        flat = torch.tensor(
            [stoi[m] for g in games for m in g.split()], dtype=torch.long
        )
    return flat, vocab, stoi


# --------------------------------------------------------------------------
# 2. Model: the same small GPT as [2.3], parameterized by vocab size
# --------------------------------------------------------------------------
class GPT(nn.Module):
    def __init__(self, vocab_size: int, block: int, d: int, heads: int, layers: int):
        super().__init__()
        self.block = block
        self.tok = nn.Embedding(vocab_size, d)
        self.pos = nn.Embedding(block, d)
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d, heads, 4 * d, batch_first=True, norm_first=True, dropout=0.0
                )
                for _ in range(layers)
            ]
        )
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab_size)

    def forward(self, idx):
        T = idx.size(1)
        h = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        causal = torch.triu(
            torch.full((T, T), float("-inf"), device=idx.device), diagonal=1
        )
        for b in self.blocks:
            h = b(h, src_mask=causal)
        return self.head(self.ln(h))


# --------------------------------------------------------------------------
# 3. THE METRIC: how many moves before the model plays something illegal?
# --------------------------------------------------------------------------
@torch.no_grad()
def legal_prefix(model, vocab, stoi, mode, device, n_games=16, temp=0.7, max_moves=60):
    """Play `n_games` from 1.e4 and return the mean number of LEGAL moves.

    This is the number that goes on your slide. It is a real, objective score:
    python-chess is the referee, not us.
    """
    model.eval()
    scores = []
    for _ in range(n_games):
        board = chess.Board()
        board.push_san("e4")
        ok = 1

        if mode == "move":
            idx = torch.tensor([[stoi["e4"]]], device=device)
            for _ in range(max_moves):
                logits = model(idx[:, -model.block :])[:, -1, :] / temp
                nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
                try:
                    board.push_san(vocab[nxt.item()])
                except Exception:
                    break  # illegal (or not a move at all) -> stop
                ok += 1
                idx = torch.cat([idx, nxt], dim=1)
        else:
            idx = torch.tensor([[stoi[c] for c in "e4 "]], device=device)
            for _ in range(max_moves * 6):  # ~6 chars per move
                logits = model(idx[:, -model.block :])[:, -1, :] / temp
                nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
                idx = torch.cat([idx, nxt], dim=1)
            line = "".join(vocab[i] for i in idx[0].tolist()).split("\n")[0]
            board, ok = chess.Board(), 0
            for tok in line.split():
                try:
                    board.push_san(tok)
                except Exception:
                    break
                ok += 1
        scores.append(ok)
    model.train()
    return sum(scores) / len(scores)


# --------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Train a chess GPT from scratch")
    p.add_argument("--tokens", choices=["move", "char"], default="move",
                   help="tokenization: one token per MOVE, or per CHARACTER")
    p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--games", type=int, default=1500)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--block", type=int, default=64)
    p.add_argument("--d-model", type=int, default=192)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--eval-every", type=int, default=200)
    args = p.parse_args()

    # `ezpz` picks the right accelerator: cuda (Polaris) / xpu (Aurora) / mps / cpu
    try:
        import ezpz
        device = ezpz.get_torch_device_type()
    except ImportError:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(0)
    games = load_games(args.games)
    flat, vocab, stoi = build_dataset(games, args.tokens)
    block = args.block if args.tokens == "move" else max(args.block, 128)

    print(f"device      : {device}")
    print(f"tokens      : {args.tokens}")
    print(f"games       : {len(games)}")
    print(f"vocab size  : {len(vocab)}")
    print(f"tokens total: {len(flat):,}")

    model = GPT(len(vocab), block, args.d_model, args.heads, args.layers).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters  : {n_params/1e6:.2f}M\n")

    flat = flat.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    history = []
    t0 = time.time()

    for step in range(1, args.steps + 1):
        ix = torch.randint(len(flat) - block - 1, (args.batch_size,), device=device)
        x = torch.stack([flat[i : i + block] for i in ix])
        y = torch.stack([flat[i + 1 : i + block + 1] for i in ix])
        loss = F.cross_entropy(model(x).reshape(-1, len(vocab)), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % args.eval_every == 0 or step == args.steps:
            lp = legal_prefix(model, vocab, stoi, args.tokens, device)
            history.append((step, loss.item(), lp))
            print(f"step {step:5d} | loss {loss.item():6.3f} | "
                  f"legal moves {lp:5.1f} | {time.time()-t0:5.0f}s")

    # ---- final: show one game the model actually played ----
    print("\nA game the model played (stopping at its first illegal move):")
    board = chess.Board()
    board.push_san("e4")
    played = ["e4"]
    if args.tokens == "move":
        idx = torch.tensor([[stoi["e4"]]], device=device)
        model.eval()
        with torch.no_grad():
            for _ in range(60):
                logits = model(idx[:, -block:])[:, -1, :] / 0.7
                nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
                try:
                    board.push_san(vocab[nxt.item()])
                except Exception:
                    break
                played.append(vocab[nxt.item()])
                idx = torch.cat([idx, nxt], dim=1)
    print("  " + " ".join(played))
    print(f"  -> {len(played)} legal moves, final FEN: {board.fen()}")

    print("\nHISTORY (step, loss, legal_moves) -- plot this:")
    for s, l, lp in history:
        print(f"  {s},{l:.4f},{lp:.2f}")


if __name__ == "__main__":
    main()
