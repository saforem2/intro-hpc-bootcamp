# Design: Shakespeare from Scratch, Entirely in `ezpz`

**Date:** 2026-07-24
**Repo:** `saforem2/intro-hpc-bootcamp`
**Status:** Approved design → implementation

## Goal

Reproduce the `wordplay` Shakespeare example
(`content/02-llms/08-shakespeare-example-colab/index.ipynb`) — a from-scratch
char-level GPT trained on tiny-Shakespeare — but **entirely in `ezpz`**, with no
`wordplay` package, no Hydra config, no external trainer or DeepSpeed. Only
dependencies: `ezpz` + `torch`.

## Key facts (researched, ezpz v0.22.1)

- ezpz ships **no** GPT/nanoGPT/Shakespeare example and **no** from-scratch
  sampler → the GPT model, char data pipeline, and generation loop are written
  **inline**.
- ezpz provides the training scaffold: `setup_torch()`, `get_torch_device()`,
  `wrap_model(model, use_fsdp=False)` (DDP; FSDP2 default; no-op at world_size≤1),
  `synchronize()` (timing), `History` (per-step `update()` → summary string;
  `finalize()` → matplotlib + terminal plots + report + jsonl/csv),
  `compute_mfu()`, `try_estimate(model, input_shape)` (FLOP count before wrap),
  `get_logger()`, `cleanup()`, `seed_everything()`.
- `sps`/`mtps` throughput are computed inline (`sps=batch/dt`,
  `mtps=batch*block/dt/1e6`); ezpz gives `loss/dt/tflops/mfu`.

## Deliverable

- **New page:** `content/02-llms/09-shakespeare-ezpz/index.qmd` (slug `09` free;
  old `09-rag-tutorial` disabled).
- **Wiring:** add to `_quarto.yml` render list + sidebar (`[02.x] Shakespeare
  from Scratch (ezpz)`), `content/index.qmd`, `content/02-llms/index.qmd`.
- House style: emoji headers, callouts, KaTeX, Authors block, Colab + GitHub
  badges.

## Structure (nanoGPT-style teaching build-up; distributed/ezpz emphasis)

1. **🎯 Setup** — `ezpz_setup .venv` + `uv pip install …/ezpz`; `rank =
   ezpz.setup_torch()`, `device = ezpz.get_torch_device()`. Callout: vs raw
   `torch.distributed` boilerplate.
2. **📖 Data** — download tiny-Shakespeare, char vocab (65), encode/decode,
   90/10 split, `get_batch()`.
3. **🧱 The GPT** — compact inline nanoGPT-style `GPT` + `GPTConfig` dataclass
   (`n_layer/n_head/n_embd/block_size/vocab_size`). Cross-link
   `00-intro-to-llms` for the attention deep-dive rather than re-teaching it.
4. **⚡ Distribute with ezpz** — `ezpz.wrap_model(model, use_fsdp=False)` (DDP);
   explain FSDP2 default + world_size≤1 no-op; `ezpz.try_estimate(...)` FLOP
   count **before** wrap.
5. **🔁 Training loop + metrics** — `ezpz.synchronize()` timing, `ezpz.History`,
   per-step `history.update({step,loss,dt,sps,mtps,tflops,mfu})` (loss/dt/tflops/
   mfu from ezpz; sps/mtps inline). Same metrics-legend table as wordplay page.
6. **💬 Generate** — inline temperature/top-k sampler; generate-before → train →
   generate-after (same arc as wordplay).
7. **📊 Finalize + plots** — `history.finalize()`; show the loss curve.
8. **🚀 Scale it up** — display-only `ezpz launch python3 train.py …` (same code,
   multi-GPU) + `## 🎒 Homework`.

## Render-safety

- **Tiny config runs at build** on CPU (`n_layer=4, n_embd=64, block_size=64,
  max_iters≈50`) — genuinely executes, real short loss curve + sample. Uses only
  torch/numpy + ezpz (all in the repo `.venv`).
- **Real 10.6M/500-iter run + multi-GPU `ezpz launch`** are display-only
  (`#| eval: false` / bash) with representative output.
- Callout: flip tiny→full config to run for real on the cluster.
- Must `quarto render` off-cluster on caddy/laptop.

## Faithfulness

Same dataset, prompt, generate→train→generate arc, and metrics legend as the
wordplay page — a true 1:1 reproduction, dependency-free.

## Verification

`quarto render content/02-llms/09-shakespeare-ezpz/index.qmd` succeeds off-cluster;
tiny training cell executes with no error; History plot renders; links resolve;
full-site render clean; committed + pushed.
