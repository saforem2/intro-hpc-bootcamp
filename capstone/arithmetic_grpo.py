"""
Capstone starter: teach a tiny model to do arithmetic with a scratchpad, by RL.
==============================================================================

This is the [3.4] RL & Reasoning recipe (GRPO with a *verifiable* reward) applied
to a task where the reward is a one-line Python check: did the model get the
right answer?

    python3 capstone/arithmetic_grpo.py              # ~3 min on a laptop CPU
    python3 capstone/arithmetic_grpo.py --steps 400  # better results

THE SETUP
    The model sees a problem like   "37+45="
    and must emit digits ending in <eos>.  Reward = 1.0 if the digits are the
    correct sum, else 0.0.  No labels, no teacher -- just a verifier.

    --scratchpad lets the model emit intermediate tokens before committing to an
    answer, i.e. "think" before answering. Compare the two: the scratchpad model
    should solve harder problems (3-digit) that the direct model cannot.

WHAT TO PUT ON YOUR SLIDE
    1. Accuracy vs training step, for --scratchpad on and off.
    2. The diversity collapse: with plain reward maximization the policy often
       converges to ONE output style. `--entropy-bonus` brings variety back.
       (Same phenomenon as the entropy-bonus section of [3.4].)
    3. A few sampled rollouts, before and after training.

Deps:  pip install torch
"""

from __future__ import annotations

import argparse
import random
import time

import torch
import torch.nn as nn


# ---- vocabulary -----------------------------------------------------------
# digits, the operators, '=' and a couple of control tokens.
DIGITS = list("0123456789")
SPECIAL = ["+", "-", "=", "|", "<eos>", "<pad>"]  # '|' separates scratchpad work
VOCAB = DIGITS + SPECIAL
STOI = {t: i for i, t in enumerate(VOCAB)}
ITOS = {i: t for t, i in STOI.items()}
EOS, PAD, SEP = STOI["<eos>"], STOI["<pad>"], STOI["|"]


def encode(s: str) -> list[int]:
    return [STOI[c] for c in s]


def decode(ids: list[int]) -> str:
    return "".join(ITOS[i] for i in ids if i not in (PAD,))


# ---- the task -------------------------------------------------------------
def make_problem(n_digits: int, rng: random.Random) -> tuple[str, int]:
    """Return ('37+45=', 82)."""
    lo, hi = 10 ** (n_digits - 1), 10**n_digits - 1
    a, b = rng.randint(lo, hi), rng.randint(lo, hi)
    return f"{a}+{b}=", a + b


def reward_fn(generated: str, answer: int) -> float:
    """THE VERIFIER. Everything after the last '|' must be the correct number.

    This is the whole supervision signal: no labels, just a Python check.
    """
    tail = generated.split("|")[-1]
    tail = tail.replace("<eos>", "").strip()
    if not tail.isdigit():
        return 0.0
    return 1.0 if int(tail) == answer else 0.0


# ---- the policy: a small GPT ----------------------------------------------
class TinyGPT(nn.Module):
    def __init__(self, vocab=len(VOCAB), d=128, heads=4, layers=3, block=48):
        super().__init__()
        self.block = block
        self.tok = nn.Embedding(vocab, d)
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
        self.head = nn.Linear(d, vocab)

    def forward(self, idx):
        T = idx.size(1)
        h = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        causal = torch.triu(
            torch.full((T, T), float("-inf"), device=idx.device), diagonal=1
        )
        for b in self.blocks:
            h = b(h, src_mask=causal)
        return self.head(self.ln(h))


# ---- GRPO -----------------------------------------------------------------
def sample_group(model, prompt_ids, k, max_new, device, temp=1.0):
    """Sample k completions for ONE prompt, tracking log-probs and entropy.

    Returns (texts, summed_logprob[k], summed_entropy[k]).
    """
    idx = torch.tensor([prompt_ids] * k, device=device)
    logps = torch.zeros(k, device=device)
    ents = torch.zeros(k, device=device)
    done = torch.zeros(k, dtype=torch.bool, device=device)

    for _ in range(max_new):
        logits = model(idx[:, -model.block :])[:, -1, :] / temp
        dist = torch.distributions.Categorical(logits=logits)
        nxt = dist.sample()
        # once a sequence has emitted <eos>, freeze it (pad, no more credit)
        logps = logps + dist.log_prob(nxt) * (~done)
        ents = ents + dist.entropy() * (~done)
        nxt = torch.where(done, torch.full_like(nxt, PAD), nxt)
        idx = torch.cat([idx, nxt[:, None]], dim=1)
        done = done | (nxt == EOS)
        if done.all():
            break

    n_prompt = len(prompt_ids)
    texts = [decode(row[n_prompt:].tolist()) for row in idx]
    return texts, logps, ents


def main():
    p = argparse.ArgumentParser(description="GRPO on arithmetic with a scratchpad")
    p.add_argument("--scratchpad", action="store_true",
                   help="let the model emit working before its answer")
    p.add_argument("--digits", type=int, default=2,
                   help="max operand size; with --curriculum we ramp up to this")
    p.add_argument("--curriculum", action="store_true",
                   help="start at 1 digit and only advance once the model is "
                        "reliable. Without this, 2-digit from scratch gets ~0 "
                        "reward forever: random rollouts essentially never hit "
                        "a 3-digit answer, so GRPO has no signal to learn from.")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--group-size", type=int, default=16,
                   help="K rollouts per prompt; the group mean is the baseline")
    p.add_argument("--prompts-per-step", type=int, default=8)
    p.add_argument("--entropy-bonus", type=float, default=0.0,
                   help="try 0.02 -- keeps output diversity from collapsing")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--eval-every", type=int, default=50)
    args = p.parse_args()

    try:
        import ezpz
        device = ezpz.get_torch_device_type()
    except ImportError:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(0)
    rng = random.Random(0)
    model = TinyGPT().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # with a scratchpad the model may emit working, then '|', then the answer
    max_new = (args.digits + 1) + (12 if args.scratchpad else 1)

    print(f"device        : {device}")
    print(f"scratchpad    : {args.scratchpad}")
    print(f"digits        : {args.digits}")
    print(f"group size K  : {args.group_size}")
    print(f"entropy bonus : {args.entropy_bonus}")
    print(f"parameters    : {sum(q.numel() for q in model.parameters())/1e6:.2f}M\n")

    @torch.no_grad()
    def evaluate(n=64, digits=None):
        """Greedy-ish accuracy on fresh problems + how many DISTINCT outputs."""
        model.eval()
        hits, outs = 0, set()
        er = random.Random(1234)
        for _ in range(n):
            prompt, ans = make_problem(digits or args.digits, er)
            txts, _, _ = sample_group(
                model, encode(prompt), 1, max_new, device, temp=0.7
            )
            hits += reward_fn(txts[0], ans)
            outs.add(txts[0])
        model.train()
        return hits / n, len(outs)

    history = []
    t0 = time.time()
    cur_digits = 1 if args.curriculum else args.digits
    recent: list[float] = []

    for step in range(1, args.steps + 1):
        total_loss = 0.0
        step_reward = 0.0
        for _ in range(args.prompts_per_step):
            prompt, ans = make_problem(cur_digits, rng)
            texts, logps, ents = sample_group(
                model, encode(prompt), args.group_size, max_new, device
            )
            rewards = torch.tensor(
                [reward_fn(t, ans) for t in texts], device=device, dtype=torch.float
            )
            step_reward += rewards.mean().item()

            # GROUP-RELATIVE advantage: the group's own mean is the baseline.
            # No value network, no reward model -- this is the whole trick.
            adv = (rewards - rewards.mean()) / (rewards.std() + 1e-6)
            loss = -(adv * logps).mean() - args.entropy_bonus * ents.mean()
            total_loss += loss

        opt.zero_grad(set_to_none=True)
        (total_loss / args.prompts_per_step).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        # curriculum: promote once the model is solidly right at this level
        recent.append(step_reward / args.prompts_per_step)
        recent = recent[-20:]
        if (args.curriculum and cur_digits < args.digits
                and len(recent) == 20 and sum(recent) / 20 > 0.60):
            cur_digits += 1
            recent = []
            print(f"  -- curriculum: advancing to {cur_digits}-digit problems --")

        if step % args.eval_every == 0 or step == args.steps:
            acc, distinct = evaluate(digits=cur_digits)
            history.append((step, step_reward / args.prompts_per_step, acc, distinct))
            print(f"step {step:4d} | {cur_digits}-digit "
                  f"| train reward {step_reward/args.prompts_per_step:.3f} "
                  f"| eval acc {acc:.3f} | distinct {distinct:3d} "
                  f"| {time.time()-t0:5.0f}s")

    print("\nSample rollouts after training:")
    er = random.Random(7)
    for _ in range(6):
        prompt, ans = make_problem(cur_digits, er)
        txt, _, _ = sample_group(model, encode(prompt), 1, max_new, device, temp=0.7)
        ok = "OK " if reward_fn(txt[0], ans) else "BAD"
        print(f"  {ok} {prompt}{txt[0]:<18} (correct: {ans})")

    print("\nHISTORY (step, train_reward, eval_acc, distinct) -- plot this:")
    for s, r, a, d in history:
        print(f"  {s},{r:.4f},{a:.4f},{d}")


if __name__ == "__main__":
    main()
