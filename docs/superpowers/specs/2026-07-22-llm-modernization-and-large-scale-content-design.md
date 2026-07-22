# Design: Modernize `ezpz` references & add Large-Scale LLM content to the Intro-to-HPC Bootcamp

**Date:** 2026-07-22
**Author:** Sam Foreman (with Claude)
**Repo:** `saforem2/intro-hpc-bootcamp-2025` (Quarto website, 2025 bootcamp content)
**Status:** Design — awaiting review before implementation planning

---

## 1. Goals

Two coordinated efforts on the bootcamp site:

1. **Modernize** stale `ezpz` (`saforem2/ezpz`) commands and API references so all content
   matches the current library (researched at `ezpz` v0.21.14, 2026-07).
2. **Add** new hands-on material to the `[02] Large Language Models` section covering:
   - Mixture of Experts (MoE), expert parallelism, DeepSeek-v3.
   - Large-scale distributed pre-training: distributing Python environments at scale
     (`ezpz yeet`), hardware/network/filesystem failures + checkpoint/restart
     (`ezpz launch --auto-retry`), throughput / MFU / efficient collectives, and
     curating data mixes / synthetic-data generation.
   - {Mid, Post}-training: continued pretraining (CPT) on high-quality tokens; SFT for
     chat/instruction-template behavior; RL (PPO, DPO, GRPO, RLVR); building an agentic
     reasoning model and the `<think>` token.

### Non-goals

- No rewrite of the conceptual intro pages (`00-intro-to-llms`, `01-hands-on-llms`) beyond
  `ezpz`-idiom checks.
- No re-teaching of parallelism primitives already covered in
  `01-neural-networks/5-distributed-training` (DDP, collectives, ZeRO, FSDP, TP, PP, SP,
  3D parallelism). New pages **reference** that page instead.
- The disabled `02-prompt-engineering` and `09-rag-tutorial` pages stay as-is (optional,
  out of scope).

---

## 2. Key decisions (locked with the user)

| Decision | Choice |
|---|---|
| Depth / format | **Full hands-on labs** that combine: ① a toy that really runs → ② real production configs/logs (display-only) → ③ a scale-up experiment. |
| Env setup | **System-agnostic** everywhere: `source <(curl -fsSL https://bit.ly/ezpz-utils) && ezpz_setup .venv`. |
| Execution target | Workshop runs on **Jupyter @ NERSC Perlmutter** (project `m4388`), but instructions stay generic (ezpz auto-detects scheduler/backend). Pages must still `quarto render` off-cluster. |
| Lab backbones | **torchtitan+ezpz** (MoE, scaling), **ezpz.examples** (yeet, `--auto-retry`, MFU, fsdp_tp/HSDP), **HF TRL/PEFT** (SFT/DPO/GRPO/PPO). Not `wordplay` for the new labs. |
| Scope | New content + full ezpz modernization + **revive/modernize `10-evaluating-llms`** + **polish `07`/`08` Shakespeare** as the pretraining on-ramp. |
| Structure | **Approach C**: five focused labs nested under an "Advanced / Large-Scale LLMs" sidebar sub-track with a short landing page. |

---

## 3. Architecture

### 3.1 Directory & numbering

```
content/02-llms/
├── index.qmd                              (EDIT: refresh contents, link advanced track)
├── 00-intro-to-llms/                      (keep; ezpz-idiom check only)
├── 01-hands-on-llms/                      (keep; ezpz-idiom check only)
├── 06-parallel-training/                  (EDIT: full ezpz modernization — main stale page)
├── 07-shakespeare-example/                (POLISH: frame as pretraining on-ramp)
├── 08-shakespeare-example-colab/          (POLISH: frame as pretraining on-ramp)
├── 10-evaluating-llms/                    (REVIVE + modernize; re-enable, order after 15)
│
├── advanced/index.qmd                     (NEW: sub-track landing / overview)
├── 11-moe/index.qmd                       (NEW: MoE & Expert Parallelism + DeepSeek-v3)
├── 12-pretraining-at-scale/index.qmd      (NEW: yeet + MFU/throughput + collectives + data)
├── 13-fault-tolerant-training/index.qmd   (NEW: failures + checkpoint/restart + --auto-retry)
├── 14-mid-post-training/index.qmd         (NEW: CPT + SFT + chat templates)
└── 15-rl-and-reasoning/index.qmd          (NEW: PPO/DPO/GRPO/RLVR + agentic <think>)
```

**Format:** all new labs are `.qmd` (matching `06`) — executable `{python}` blocks plus
KaTeX, `::: {#fig-*}` crossrefs, mermaid, and callouts. `.ipynb` stays only where a
Colab-first experience is central (existing `08`).

### 3.2 Shared "toy → real → scale" anatomy

Every new lab has the same three movements:

| Movement | What it is | Executes at render? |
|---|---|---|
| **① Toy that runs** | Minimal real example on 1–few devices (CPU fallback via `ezpz.get_torch_device()`), tiny shapes; genuinely executes and is cached via Quarto `freeze`. | ✅ yes |
| **② Real configs** | Annotated, display-only production configs + real log/metric excerpts (e.g. DeepSeek-v3 MoE config; a 400B-scale MFU log). | ⬜ display-only |
| **③ Scale-up experiment** | Guided "launch it wider" step using `ezpz launch [--auto-retry]`, plus homework to submit proof of a run (mirrors `06`'s homework pattern). | ▶️ student-run (`#| eval: false`) |

Shared callout vocabulary: `::: {.callout-tip title="🔬 At DeepSeek-v3 scale…"}` for
concept→scale bridges; `::: {.callout-warning title="⏰ …"}` reused from the `[01]` style;
`::: {.callout-note title="Authors"}` attribution block on every page.

### 3.3 Wiring (each new/revived page registered in three places)

1. **`content/_quarto.yml`** — add to `project.render` list **and** to
   `website.sidebar.contents` under a new nested "Advanced / Large-Scale LLMs" section entry.
   `10-evaluating-llms` re-enabled and ordered after `15`.
2. **`content/index.qmd`** — add to the `[02] Large Language Models` project-contents bullets.
3. **`README.md`** — refresh setup snippet(s) to the current `ezpz_setup .venv` one-liner.

---

## 4. Per-lab content detail

### `advanced/index.qmd` — Sub-track landing
Overview of what "large-scale" adds beyond `[01]`; a map of the five labs; the one-time
env setup; a "how to run on Perlmutter Jupyter / any ALCF system" callout; a back-link to
`[01] Distributed Training` for assumed primitives.

### `11-moe/` — Mixture of Experts & Expert Parallelism · torchtitan+ezpz
- **①** Small transformer with a real top-k router + a few experts (torchtitan MoE at tiny
  scale, or a from-scratch `MoELayer`), printing token routing, aux load-balancing loss,
  and per-expert utilization.
- **②** Annotated DeepSeek-v3 architecture (256 routed + 1 shared expert, MLA,
  aux-loss-free balancing, MTP) as display-only config + diagram; dense-vs-MoE FLOP/param
  contrast.
- **③** `ezpz launch` a torchtitan MoE with `--parallelism.expert_parallel_degree N`;
  mermaid diagram of expert-parallel all-to-all vs. DP/TP/PP.
- Concepts: sparse activation, router, load balancing, capacity factor, EP vs TP/DP, all-to-all.

### `12-pretraining-at-scale/` — ezpz.examples + torchtitan
- **①** `ezpz yeet` a venv/dataset tarball to node-local `/tmp`, launch from it, measure MFU
  via `ezpz.flops.compute_mfu()` / `--profile`.
- **②** Production `tokens/sec` + `mfu` log excerpt; a data-mixture (domain-weight) table;
  a short synthetic-data-generation recipe (display-only).
- **③** Node-count sweep, plot throughput scaling + MFU; discuss efficient collectives
  (NCCL/oneCCL/torchcomms), overlap, and why `yeet` beats NFS import-storms at scale.
- Concepts: env distribution, MFU/throughput, collective efficiency, data curation/mixes,
  synthetic data.

### `13-fault-tolerant-training/` — ezpz.examples
- **①** Real checkpoint → kill a rank → restart-from-checkpoint that recovers, in a
  Jupyter-runnable loop.
- **②** Failure taxonomy (network flaps, shared-FS stalls, stragglers/dead nodes, OOM) with
  real bad-node log signatures (`ezpz.failover` patterns).
- **③** `ezpz launch --auto-retry --spare-nodes auto` + the idle-output `--timeout`
  watchdog; explain the failover loop + termination conditions.
- Concepts: checkpoint/restart, DCP, `--auto-retry`, spare nodes, watchdog timeouts,
  shared-filesystem pitfalls at scale.

### `14-mid-post-training/` — CPT + SFT · HF TRL/PEFT
- **①** CPT on a small high-quality corpus, then SFT with `SFTTrainer` + a chat template on
  a base model (LoRA/PEFT so it fits one node).
- **②** What "high-quality tokens" means for CPT; anatomy of a chat/instruction template
  (system/user/assistant, special tokens); a real SFT dataset schema.
- **③** Multi-GPU SFT via `ezpz launch`; homework: fine-tune to follow a custom instruction
  template.
- Concepts: continued pretraining, SFT, chat templates, instruction tuning, LoRA/PEFT.

### `15-rl-and-reasoning/` — HF TRL
- **①** A GRPO/RLVR loop on a small model with a verifiable reward (e.g. math/format
  correctness); a minimal DPO preference example.
- **②** PPO vs DPO vs GRPO vs RLVR (table); how RLVR removes the reward model; the
  reasoning-model recipe (base → SFT → RLVR).
- **③** Build an agentic reasoning model: train/prompt for the `<think>…</think>` pattern,
  show a rollout; homework to reward-shape reasoning traces.
- Concepts: PPO, DPO, GRPO, RLVR, reward models vs verifiers, agentic reasoning, `<think>`.

---

## 5. `ezpz` modernization sweep

Grounded in the confirmed staleness map (see `memory/ezpz-modernization.md`). All shown
commands must match the current CLI; no invented flags.

**Stale → current mapping:**
- `python3 -m ezpz.test_dist` / `mpirun -n $NGPUS python3 -m ezpz.test_dist` → `ezpz test`
  (or `ezpz launch python3 -m ezpz.examples.test`).
- `ezpz.dist` → `ezpz.distributed`; `ezpz.setup()` → `ezpz.setup_torch()`.
- `ezpz-launch` / `ezpz-test` / `ezpz-yeet-env` → `ezpz launch` / `ezpz test` / `ezpz yeet`.
- `mpirun -n $NGPUS python3 …` → `ezpz launch python3 …` (auto-detects scheduler).
- `source deps/ezpz/src/ezpz/bin/utils.sh` + `ezpz_setup_python`/`ezpz_setup_job`
  → `source <(curl -fsSL https://bit.ly/ezpz-utils) && ezpz_setup .venv`.
- install → `uv pip install git+https://github.com/saforem2/ezpz`.

**Files to sweep:**
- `content/02-llms/06-parallel-training/index.qmd` — **main offender**: old utils.sh path,
  `ezpz_setup_job`, `mpirun … ezpz.test_dist`, `mpirun … wordplay`, `ALCFAITP` allocation,
  Sophia-only proxy block (generalize into a per-system callout), 2024 date.
- `content/index.qmd` — `ezpz-test` → `ezpz test`, `ezpz-launch -m wordplay` →
  `ezpz launch -m wordplay` (keep the working W&B run output).
- `README.md` — align both setup snippets to `ezpz_setup .venv`.
- Also grep-sweep: `content/00-intro-AI-HPC/{5-mcmc-example,6-linear-regression}`,
  `content/01-neural-networks/5-distributed-training`,
  `content/02-llms/07-shakespeare-example/wordplay/index.qmd`
  (old `savejobenv` / `launch`-alias / `setup_conda_polaris` idioms).

A grep-driven checklist (search for `ezpz.test_dist`, `ezpz.dist`, `ezpz-launch`,
`ezpz-test`, `deps/ezpz`, `ezpz_setup_job`, `mpirun -n`) ensures none are missed.

---

## 6. Revive / polish existing pages

- **`10-evaluating-llms/`** — rewrite the 2024 link-stub into a real house-style page
  (LLM evaluation, benchmarks, pitfalls, verifiers-tie-in to RLVR); re-enable in
  `_quarto.yml` + sidebar, ordered **after** `[15]`.
- **`07`/`08` Shakespeare** — light polish framing them as the from-scratch pretraining
  on-ramp to the advanced track (forward-link callout + date refresh + ezpz-idiom check).
  No structural rewrite.

---

## 7. Build, testing, error handling

- **Rendering off-cluster (authoring rule):** the site must `quarto render` on any machine
  (laptop/CI) without GPUs, a scheduler, or heavy libs (torchtitan/trl/deepspeed) installed.
  Therefore:
  - Cells that require a GPU, a scheduler, or a not-in-`.venv` library are marked
    `#| eval: false` (they render as highlighted, copy-pasteable code but do not execute at
    build time). ③ scale steps use display-only ```` ```bash ```` fences.
  - Only cells using libraries already in the repo `.venv` (torch [CPU ok], numpy,
    matplotlib, rich, transformers-for-tiny-models) and tiny shapes may stay executable
    (`eval: true`, the default), degrading to CPU via `ezpz.get_torch_device()`.
  - The "toy that really runs" guarantee is honored in the **workshop environment**
    (Perlmutter Jupyter, where the heavy libs + GPUs exist); `eval: false` only governs the
    static site build. Where practical, ① cells are written so a student can flip
    `eval: false`→`true` and run them as-is on the cluster.
- **Verification (evidence before claiming done):** `quarto render content` must succeed
  with all new pages enabled; sidebar/nav renders; internal links resolve; the ezpz-sweep
  grep checklist returns clean.
- **Correctness guards:** every ezpz command matches the confirmed current CLI; commands
  that can't execute at render time are display-only and labeled with their target system.
  Placeholders use angle brackets (`-A <project>`), not stale real values.
- **Isolation:** each lab is a self-contained directory (own `index.qmd`, own `images/`),
  independently renderable and teachable; cross-references are links only; no shared
  mutable state.

---

## 8. Rollout order (for the implementation plan)

1. **ezpz modernization sweep + wiring scaffold** (low-risk, immediately shippable).
2. **`advanced/` landing + sidebar grouping.**
3. **The five labs (11→15)**, each authored toy-first so it renders before real configs
   are added.
4. **Revive `10`; polish `07`/`08`.**

---

## 9. Open items to confirm during review

- Exact NERSC/Perlmutter allocation string to show as the concrete example (vs. a generic
  `<project>` placeholder). Currently planned as a placeholder with an `m4388` note.
- Whether any lab warrants an `.ipynb` (Colab badge) companion in addition to the `.qmd`,
  as `07`/`08` have.
