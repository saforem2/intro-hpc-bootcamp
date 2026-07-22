# Intro to HPC Bootcamp
Sam Foreman
2025-07-22

<link rel="preconnect" href="https://fonts.googleapis.com">

- [🚀 Parallel Training Methods for
  AI](#rocket-parallel-training-methods-for-ai)
  - [👋 Hands On](#wave-hands-on)
  - [🎒 Homework](#school_satchel-homework)

# 🚀 Parallel Training Methods for AI

[Sam Foreman](https://samforeman.me) [Intro to AI-driven Science on
Supercomputers](https://www.alcf.anl.gov/alcf-ai-science-training-series)
*2026-07-22*

[![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/saforem2/intro-hpc-bootcamp/blob/main/docs/02-llms/06-parallel-training/index.ipynb)
[![](https://img.shields.io/badge/-View%20on%20GitHub-333333?style=flat&logo=github&labelColor=gray.png)](https://github.com/saforem2/intro-hpc-bootcamp/blob/main/content/02-llms/06-parallel-training/index.qmd)

- Slides: <https://samforeman.me/talks/ai-for-science-2024/slides>
  - HTML version: <https://samforeman.me/talks/ai-for-science-2024>

## 👋 Hands On

> [!NOTE]
>
> This workshop runs on [NERSC
> Perlmutter](https://docs.nersc.gov/systems/perlmutter/) (SLURM: use
> `salloc` / Jupyter). The `ezpz launch` command below is
> scheduler-agnostic — it auto-detects the environment (PBS → `mpiexec`,
> SLURM → `srun`) — so the same steps work on ALCF PBS systems too.

> [!NOTE]
>
> ### Proxies on ALCF login nodes
>
> On ALCF PBS systems (e.g. Sophia, Polaris) the login/compute nodes
> need proxy settings for outbound network access — run these before
> cloning/installing:
>
> ``` bash
> export HTTP_PROXY="http://proxy.alcf.anl.gov:3128"
> export HTTPS_PROXY="http://proxy.alcf.anl.gov:3128"
> export http_proxy="http://proxy.alcf.anl.gov:3128"
> export https_proxy="http://proxy.alcf.anl.gov:3128"
> export ftp_proxy="http://proxy.alcf.anl.gov:3128"
> ```

1.  Submit interactive job (PBS example; on Perlmutter use `salloc`
    instead):

    ``` bash
    qsub -A <project> -q by-node -l select=1 -l walltime=01:00:00,filesystems=eagle:home -I
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
- A path to a logfile containing the output from a run on the ALCF
  filesystems
- A screenshot of:
  - the text printed out from the run
  - a graph from the W&B Run
  - anything that shows that you clearly were able to run the example
- url to a W&B Run or [W&B
  Report](https://api.wandb.ai/links/aurora_gpt/7du35js1)
- etc.
