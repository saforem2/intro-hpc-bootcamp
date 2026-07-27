# Intro to HPC Bootcamp
Sam Foreman
2025-07-22

<link rel="preconnect" href="https://fonts.googleapis.com">

- [🚀 Hands-On: Launching a Distributed Training
  Run](#rocket-hands-on-launching-a-distributed-training-run)
  - [👋 Hands On](#wave-hands-on)
  - [🎒 Homework](#school_satchel-homework)

# 🚀 Hands-On: Launching a Distributed Training Run

[Sam Foreman](https://samforeman.me) [Intro to AI-driven Science on
Supercomputers](https://www.alcf.anl.gov/alcf-ai-science-training-series)
*2026-07-22*

[![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/saforem2/intro-hpc-bootcamp/blob/main/docs/02-llms/1-parallel-training/index.ipynb)
[![](https://img.shields.io/badge/-View%20on%20GitHub-333333?style=flat&logo=github&labelColor=gray.png)](https://github.com/saforem2/intro-hpc-bootcamp/blob/main/content/02-llms/1-parallel-training/index.qmd)

- Slides: <https://samforeman.me/talks/ai-for-science-2024/slides>
  - HTML version: <https://samforeman.me/talks/ai-for-science-2024>

This is a **hands-on lab**: you’ll launch a real multi-GPU
[data-parallel](#concepts-recap) training run on a supercomputer. It’s
the practical counterpart to the [**\[01\] Distributed
Training**](../../01-neural-networks/4-distributed-training/index.qmd)
lesson — read that first if the concepts below are new to you.

> [!NOTE]
>
> ### 🧠 Concepts recap (30-second version)
>
> The recipe below runs **Distributed Data Parallelism (DDP)**, the most
> common way to scale training:
>
> - **Data Parallel (DP/DDP)** — every GPU holds a *full copy* of the
>   model and processes a different slice of each batch; gradients are
>   averaged across GPUs with an **all-reduce** so every replica stays
>   in sync. This is what you launch here.
> - **Tensor Parallel (TP)** — a single layer’s matrices are split
>   *across* GPUs (usually within one node). Used when a layer is too
>   big for one GPU.
> - **Pipeline Parallel (PP)** — different *layers* live on different
>   GPUs and micro-batches flow through the stages like an assembly
>   line.
>
> Real large-scale LLM training combines all three. For the full
> treatment — all-reduce/broadcast/gather, ZeRO/FSDP, and
> pipeline/tensor parallelism with runnable examples — see [**\[01\]
> Distributed
> Training**](../../01-neural-networks/4-distributed-training/index.qmd).

## 👋 Hands On

> [!NOTE]
>
> This workshop runs on [NERSC
> Perlmutter](https://docs.nersc.gov/systems/perlmutter/) (SLURM,
> project **`m4388`**). The `ezpz launch` command below is
> scheduler-agnostic — it auto-detects the environment (SLURM → `srun`,
> PBS → `mpiexec`) — so the same steps also work on ALCF PBS systems if
> you switch the allocation command.

1.  Log in to Perlmutter and request an interactive GPU node:

    ``` bash
    ssh <user>@perlmutter.nersc.gov
    salloc --nodes 1 --qos interactive --time 01:00:00 --constraint gpu --account m4388
    ```

2.  Clone [`saforem2/wordplay`](https://github.com/saforem2/wordplay):

    ``` bash
    git clone https://github.com/saforem2/wordplay
    cd wordplay
    ```

3.  Setup python (creates / activates a `.venv` and detects the job):

    ``` bash
    source <(curl -fsSL https://bit.ly/ezpz-utils) && ezpz_setup .venv
    ```

4.  Install `{ezpz, wordplay}`:

    ``` bash
    uv pip install git+https://github.com/saforem2/ezpz
    uv pip install -e .
    ```

5.  Setup (or disable) [`wandb`](https://wandb.ai):

    ``` bash
    # to setup:
    wandb login
    # to disable:
    export WANDB_DISABLED=1
    ```

6.  Test Distributed Setup:

    ``` bash
    ezpz test
    ```

    See:
    [`ezpz/test_dist.py`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/test_dist.py)

7.  Prepare Data:

    ``` bash
    python3 data/shakespeare_char/prepare.py
    ```

8.  Launch Training:

    ``` bash
    ezpz launch python3 -m wordplay \
        train.backend=DDP \
        train.eval_interval=100 \
        data=shakespeare \
        train.dtype=bf16 \
        model.batch_size=64 \
        model.block_size=1024 \
        train.max_iters=1000 \
        train.log_interval=10 \
        train.compile=false
    ```

## 🎒 Homework

Submit *proof* that you were able to successfully follow the above
instructions and launch a distributed data parallel training run.

Where *proof* can be any of:

- The contents printed out to your terminal during the run
- A path to a logfile containing the output from a run on the NERSC
  filesystems (e.g. under your `m4388` project space)
- A screenshot of:
  - the text printed out from the run
  - a graph from the W&B Run
  - anything that shows that you clearly were able to run the example
- url to a W&B Run or [W&B
  Report](https://api.wandb.ai/links/aurora_gpt/7du35js1)
- etc.
