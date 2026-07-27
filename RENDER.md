# Rendering the site (and the GPU pages)

The site is a [Quarto](https://quarto.org) project: pages live in `content/**/index.qmd`
and the built HTML is committed to `docs/` (served by GitHub Pages). Most pages
render fine on a laptop. A **few pages want a real GPU** — this doc explains how
to render those on a cluster (Polaris / Sunspot / Aurora / Perlmutter) so their
outputs get baked into the published site.

## The key idea: render where the compute is, commit the output

Quarto **executes** the code cells at render time and writes the results straight
into `docs/`. So whichever machine renders a page determines what the reader sees
— and the reader never runs anything themselves; they just view the committed
HTML.

That means the workflow for a GPU page is simply:

1. Clone the repo **on the cluster**, inside a GPU allocation.
2. Render just that page there — Quarto runs its cells on the GPU.
3. Commit the resulting `docs/<page>/` output and push.

No reader needs a GPU. No CI re-renders (there is none — `docs/` is hand-committed),
so a cluster render is never clobbered.

> **On `content/_freeze/`:** Quarto also caches executed outputs under
> `content/_freeze/`. We keep that directory **untracked** (it's in `.gitignore`)
> — the source of truth we publish is the committed `docs/` output, not the freeze
> cache. `render-on-cluster.sh` deletes a page's freeze before rendering so the GPU
> genuinely re-executes the cells instead of replaying a stale CPU cache.

## Which pages want a GPU

Run `scripts/render-on-cluster.sh --list`. Currently:

| Page | Why a GPU helps |
|------|-----------------|
| `01-neural-networks/3-representation-learning` | contrastive pretraining is only compelling at scale (big batches, more epochs) |
| `01-neural-networks/4-distributed-training` | DDP / collective ops / genuinely multi-GPU |
| `02-llms/1-parallel-training` | the `wordplay` DDP launch |
| `02-llms/3-shakespeare-ezpz` | a real GPT training run |
| `03-ai-for-science/0-genslm` | genome LM — larger runs than the CPU toy |

Everything else renders on a laptop/CI and should stay that way.

## Step-by-step (NERSC Perlmutter — SLURM)

```bash
# 1. Get an interactive GPU allocation
ssh perlmutter
NODES=1; salloc --nodes $NODES --qos interactive --time 01:00:00 -C gpu \
  --gpus=$(( 4 * NODES )) -A m4388_g

# 2. Clone + set up the environment
git clone https://github.com/saforem2/intro-hpc-bootcamp
cd intro-hpc-bootcamp
source <(curl -fsSL https://bit.ly/ezpz-utils) && ezpz_setup .venv
# (installs torch etc. into .venv; see README.md for the full dependency list)

# 3. Render a GPU page (executes its cells on the GPU)
scripts/render-on-cluster.sh 01-neural-networks/4-distributed-training
#   ... or render the whole GPU set:
scripts/render-on-cluster.sh --all-gpu

# 4. Commit just that page's built output and push
git add docs/01-neural-networks/4-distributed-training/
git commit -m "docs: GPU render of distributed-training on $(hostname)"
git push
```

## Step-by-step (ALCF Polaris / Sunspot — PBS)

Same idea; only the allocation command and modules differ:

```bash
# Polaris (PBS) — interactive GPU node
qsub -I -A <project> -q debug -l select=1 -l walltime=01:00:00 \
     -l filesystems=home:eagle

git clone https://github.com/saforem2/intro-hpc-bootcamp
cd intro-hpc-bootcamp
module load conda; conda activate base          # or your course env
source <(curl -fsSL https://bit.ly/ezpz-utils) && ezpz_setup .venv

scripts/render-on-cluster.sh 02-llms/3-shakespeare-ezpz
git add docs/02-llms/3-shakespeare-ezpz/ && git commit -m "docs: GPU render on $(hostname)" && git push
```

(`ezpz` auto-detects the scheduler, so the in-page `ezpz launch` commands work on
both SLURM and PBS. See `content/00-intro-AI-HPC/2-jupyter-notebooks/` for the
PBS↔SLURM cheat-sheet.)

## Notes

- **Render one page at a time.** `quarto render content/<page>/index.qmd` only
  touches that page's `docs/` output; sibling pages are left untouched, so a GPU
  render won't disturb the CPU-rendered pages.
- **Commit only the page you rendered.** `git add docs/<page>/` (plus
  `docs/search.json` / `docs/sitemap.xml` if they changed) — not the whole
  `docs/` tree — to keep the diff focused.
- **The script prints the GPU it sees** before rendering; if it warns "no CUDA/XPU
  visible," you're not inside an allocation and the cells will fall back to CPU.
- Building the *whole* site (CPU) is still just `quarto render content` from a
  laptop, as in `README.md`.
