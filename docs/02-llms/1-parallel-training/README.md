# 🚀 Hands-On: Launching a Distributed Training Run
Sam Foreman
2026-07-22

<link rel="preconnect" href="https://fonts.googleapis.com">

- [👋 Hands On](#wave-hands-on)
- [🎒 Homework](#school_satchel-homework)

> [!NOTE]
>
> ### Authors
>
> Written by [Sam Foreman](https://samforeman.me) for the [Intro to
> AI-driven Science on
> Supercomputers](https://www.alcf.anl.gov/alcf-ai-science-training-series)
> / [Intro to HPC Bootcamp](https://intro-hpc-bootcamp.alcf.anl.gov/).

[![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/saforem2/intro-hpc-bootcamp/blob/main/docs/02-llms/1-parallel-training/index.ipynb)
[![](https://img.shields.io/badge/-View%20on%20GitHub-333333?style=flat&logo=github&labelColor=gray.png)](https://github.com/saforem2/intro-hpc-bootcamp/blob/main/content/02-llms/1-parallel-training/index.qmd)

- Slides: <https://samforeman.me/talks/ai-for-science-2024/slides>
  - HTML version: <https://samforeman.me/talks/ai-for-science-2024>

This is a **hands-on lab**: you’ll launch a real multi-GPU
[data-parallel](#concepts-recap) training run on a supercomputer. It’s
the practical counterpart to the [**\[1\] Distributed
Training**](../../01-neural-networks/4-distributed-training/index.qmd)
lesson. Read that first if the concepts below are new to you.

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
> treatment (all-reduce/broadcast/gather, ZeRO/FSDP, and pipeline/tensor
> parallelism with runnable examples), see [**\[1\] Distributed
> Training**](../../01-neural-networks/4-distributed-training/index.qmd).

## 👋 Hands On

> [!NOTE]
>
> This workshop runs on [NERSC
> Perlmutter](https://docs.nersc.gov/systems/perlmutter/) (SLURM,
> project **`m4388`**). The `ezpz launch` command below is
> scheduler-agnostic: it auto-detects the environment (SLURM → `srun`,
> PBS → `mpiexec`), so the same steps also work on ALCF PBS systems if
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
    source <(curl -fsSL https://bit.ly/ezpz-utils) && ezpz_setup_env
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

> [!TIP]
>
> ### ✅ Confirm it is really multi-GPU
>
> Before launching anything expensive, make sure `ezpz` actually
> discovered **more than one GPU**. Early in the output, `ezpz` prints a
> distributed-setup banner with one line per rank, something like:
>
> ``` console
> [2026-07-22 ...][INFO][ezpz.dist] - Using device='cuda' with backend='nccl'
> [2026-07-22 ...][INFO][ezpz.dist] - ['host'][0/3]   # rank 0 of 4
> [2026-07-22 ...][INFO][ezpz.dist] - ['host'][1/3]   # rank 1 of 4
> [2026-07-22 ...][INFO][ezpz.dist] - ['host'][2/3]   # rank 2 of 4
> [2026-07-22 ...][INFO][ezpz.dist] - ['host'][3/3]   # rank 3 of 4
> ```
>
> The number to check is **`WORLD_SIZE`**: the total number of ranks
> (GPUs) across all nodes. Look for the `[i/N]` rank tags above (here
> `N+1 = 4`), or print it directly:
>
> ``` bash
> echo "WORLD_SIZE=${WORLD_SIZE}"   # set by the launcher inside the job
> ```
>
> - **`WORLD_SIZE` \> 1** → you have data parallelism. 🎉 Every rank
>   above is a separate GPU that will process a different slice of each
>   batch.
> - **`WORLD_SIZE` == 1** (only a single `[0/0]` rank) → you’re on a
>   **1-GPU** allocation and are *not* doing data-parallel training. Go
>   back and request more GPUs (e.g. `salloc ... --gpus-per-node 4`, or
>   add `--nodes`) before continuing.

1.  Prepare Data:

    ``` bash
    python3 data/shakespeare_char/prepare.py
    ```

2.  Launch Training:

    ``` bash
    ezpz launch python3 -m wordplay \
        train.backend=DDP \
        train.eval_interval=100 \
        data=shakespeare \
        train.dtype=bfloat16 \
        model.batch_size=64 \
        model.block_size=1024 \
        train.max_iters=1000 \
        train.log_interval=10 \
        train.compile=false
    ```

> [!TIP]
>
> ### ✅ A healthy training run
>
> `ezpz launch` first re-prints the distributed banner (again, confirm
> `WORLD_SIZE` matches the number of GPUs you requested), then starts
> logging one line every `log_interval` (=10) steps. The key signal of
> success is that the **`loss` trends downward** as `iter` climbs:
>
> ``` console
> [ezpz.dist] - Using device='cuda' with backend='nccl', WORLD_SIZE=4
> ...
> [wordplay] - iter=10   loss=3.1420  dt=0.182s  sps=5.49  mtps=0.36  mfu=18.7%
> [wordplay] - iter=20   loss=2.8107  dt=0.179s  sps=5.58  mtps=0.37  mfu=19.1%
> [wordplay] - iter=50   loss=2.4013  dt=0.178s  sps=5.61  mtps=0.37  mfu=19.3%
> [wordplay] - iter=100  loss=2.0459  dt=0.177s  sps=5.64  mtps=0.37  mfu=19.4%
> [wordplay] - iter=100  eval: train_loss=2.03  val_loss=2.11
> ...
> ```
>
> (Exact numbers depend on the machine and batch size, so don’t worry
> about matching them.) What matters:
>
> - **`loss` is going *down*** over the first ~100 iters (from ~3–4
>   toward ~2 or lower on tiny-Shakespeare). A flat or rising loss means
>   something is wrong.
> - **`dt`** (seconds/iter) is roughly constant — steady throughput.
> - The periodic **`eval`** line (every `eval_interval`=100 iters)
>   reports `train_loss` / `val_loss`; both should be falling early on.
>
> If instead you see a single rank, a `loss` stuck near its initial
> value, or NCCL/`srun` errors, stop and re-check the earlier steps.

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
