# Capstone starters

Two self-contained projects that build on what the bootcamp already covered.
Both run on a laptop CPU in a few minutes, both scale up with `ezpz launch`, and
both produce an **objective number** you can put on a slide — not a vibe check.

Every number below was measured on the committed code. Reproduce them first, then
go beyond them.

```bash
pip install torch python-chess     # already present in the shared NERSC env
```

---

## 1. `chess_gpt.py` — train a GPT to play legal chess

Same model as [\[2.3\] Shakespeare from Scratch](../content/02-llms/3-shakespeare-ezpz/);
only the data and the metric change. Trains on 1500 real tournament games.

**The metric:** generate a game from `1.e4` and ask `python-chess` how many moves
the model plays before it produces an illegal one.

```bash
python3 capstone/chess_gpt.py                    # move-level tokens (default)
python3 capstone/chess_gpt.py --tokens char      # the comparison
python3 capstone/chess_gpt.py --steps 3000       # go further
```

**The finding to present** (measured, ~40 s each on a laptop):

| tokenization | final loss | legal moves |
|---|---|---|
| character-level | **1.24** (lower!) | 1.4 |
| move-level | 1.96 | **16.8** |

The character model gets the *better loss* and plays *worse chess*. It learns
what SAN looks like — `Nf3`, `Bxc4+` — without any idea where the pieces are, so
it emits well-formed illegal moves forever. One token per move lets it model the
game instead of the spelling.

That gap is the whole talk: **your loss curve is not your metric.** Almost every
real ML failure looks like this.

After 600 steps the move-level model opens with real theory:

```
e4 e6 d4 d5 e5 c5 c3 Bd7 Bd3      <- French Defence, Advance Variation
```

**Where to take it**
- Plot legal-move count vs. training step for both tokenizations on one axis.
- Does a bigger model help, or does it need more *games*? (Try `--games 400` vs `4000`.)
- Where does it break — opening, middlegame, or when castling/en-passant come up?
- Train on one player's games (Morphy, Tal) and see whether the openings shift.

---

## 2. `arithmetic_grpo.py` — teach a model to add, with RL

The [\[3.4\] RL & Reasoning](../content/03-advanced-llms/4-rl-and-reasoning/)
recipe on a task where the reward is a one-line Python check. No labels, no
reward model, no teacher — just a verifier:

```python
def reward_fn(generated, answer):
    return 1.0 if int(tail) == answer else 0.0
```

```bash
python3 capstone/arithmetic_grpo.py --digits 1 --steps 150          # learns
python3 capstone/arithmetic_grpo.py --curriculum --digits 2 --steps 250
python3 capstone/arithmetic_grpo.py --curriculum --digits 2 --steps 250 \
        --entropy-bonus 0.02                                        # the A/B
```

**Finding 1 — cold start.** Two-digit addition from scratch gets a reward of
**~0.00 forever**. Random rollouts essentially never land on the right 3-digit
answer, so GRPO has no signal to learn from. `--curriculum` starts at 1 digit and
promotes only once the model is reliable. Worth showing: *RL needs a reachable
reward, or it does not start at all.*

**Finding 2 — diversity collapse and the fix.** Same run, 250 steps, only the
entropy bonus differs:

| | train reward | eval accuracy | distinct outputs |
|---|---|---|---|
| plain GRPO | 0.24 | 0.078 | **3** — it just answers "9" every time |
| `--entropy-bonus 0.02` | **0.49** | **0.250** | **12** |

Without the bonus the policy finds one output that sometimes scores and stops
exploring — textbook reward hacking, reproduced on a 0.6M-parameter model in
about a minute. This is the same phenomenon as the entropy-bonus section of
\[3.4\], and you can generate the plot yourself.

**Where to take it**
- Sweep `--entropy-bonus` (0, 0.01, 0.02, 0.05) and plot accuracy *and* diversity.
  Too much entropy and the model never commits — find the knee.
- Does `--scratchpad` (letting it emit working before the answer) beat direct
  answering at 2–3 digits?
- Swap the verifier for a different task: strict JSON output, sorting a list,
  the 24 game. The GRPO loop does not change at all — only `reward_fn` does.
  **That is the point of verifiable rewards.**

---

## Running on Perlmutter

```bash
salloc --nodes 1 --qos interactive --time 01:00:00 -C gpu --gpus 4 -A m4388_g
module load conda
conda activate /global/cfs/cdirs/m4388/envs/intro-hpc-bootcamp

python3 capstone/chess_gpt.py --steps 5000       # single GPU
ezpz launch python3 capstone/chess_gpt.py        # all GPUs in the job
```

Both scripts call `ezpz.get_torch_device_type()`, so they pick up `cuda` on
Perlmutter/Polaris, `xpu` on Aurora, `mps` on a Mac, and fall back to CPU —
no code changes needed.
