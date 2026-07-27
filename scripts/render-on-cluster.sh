#!/usr/bin/env bash
#
# render-on-cluster.sh — render one (or all) bootcamp page(s) on a GPU machine
# (Polaris / Sunspot / Aurora / Perlmutter) so their executed outputs are baked
# into docs/. See RENDER.md for the full workflow and rationale.
#
# The idea: a few pages want a real GPU (distributed training, a genuine model
# run). Render those pages *here*, on the cluster, from inside an allocation.
# Quarto executes the cells against the GPU and writes the results into docs/
# (and content/_freeze/). You then commit the resulting docs/ output; the
# published site just serves it, so no reader needs a GPU.
#
# Usage:
#   scripts/render-on-cluster.sh <page-path> [<page-path> ...]
#   scripts/render-on-cluster.sh 01-neural-networks/4-distributed-training
#   scripts/render-on-cluster.sh --list        # show GPU-relevant pages
#   scripts/render-on-cluster.sh --all-gpu     # render the whole GPU set
#
# Run this from INSIDE a GPU allocation (see RENDER.md for the salloc/qsub line).
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Pages that genuinely benefit from a GPU (edit as the course evolves).
GPU_PAGES=(
  "01-neural-networks/3-representation-learning"   # contrastive pretraining at scale
  "01-neural-networks/4-distributed-training"      # DDP / collectives / multi-GPU
  "02-llms/1-parallel-training"                     # wordplay DDP launch
  "02-llms/3-shakespeare-ezpz"                      # real GPT training run
  "03-ai-for-science/0-genslm"                      # genome LM (larger runs)
)

usage() { sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

case "${1:-}" in
  ""|-h|--help) usage 0 ;;
  --list)
    echo "GPU-relevant pages:"
    printf '  %s\n' "${GPU_PAGES[@]}"
    exit 0 ;;
  --all-gpu) PAGES=("${GPU_PAGES[@]}") ;;
  *) PAGES=("$@") ;;
esac

# --- environment ---------------------------------------------------------------
# Prefer an already-activated venv; otherwise fall back to the repo .venv, and
# point Quarto at that interpreter so cells run against the GPU-enabled Python.
if [[ -z "${VIRTUAL_ENV:-}" && -x "$REPO_ROOT/.venv/bin/python" ]]; then
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.venv/bin/activate"
fi
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "!! No virtualenv active and no $REPO_ROOT/.venv found."
  echo "   Set one up first (see RENDER.md), e.g.:"
  echo "     source <(curl -fsSL https://bit.ly/ezpz-utils) && ezpz_setup .venv"
  exit 1
fi
export QUARTO_PYTHON="${QUARTO_PYTHON:-$VIRTUAL_ENV/bin/python}"

echo "== repo:   $REPO_ROOT"
echo "== python: $QUARTO_PYTHON"
"$QUARTO_PYTHON" - <<'PY' || true
import torch
print(f"== torch:  {torch.__version__}")
if torch.cuda.is_available():
    print(f"== CUDA:   {torch.cuda.device_count()} GPU(s) -> {torch.cuda.get_device_name(0)}")
else:
    try:
        import intel_extension_for_pytorch as ipex  # noqa: F401
        n = torch.xpu.device_count()
        print(f"== XPU:    {n} device(s)")
    except Exception:
        print("== WARNING: no CUDA/XPU visible — are you inside a GPU allocation?")
PY

# --- render --------------------------------------------------------------------
# Deleting the page's _freeze forces a fresh execution on THIS machine (so the
# GPU actually runs the cells rather than replaying a stale CPU cache).
for page in "${PAGES[@]}"; do
  page="${page%/}"; page="${page#content/}"     # normalize
  src="content/$page/index.qmd"
  if [[ ! -f "$src" ]]; then
    echo "!! no such page: $src (skipping)"; continue
  fi
  echo ""
  echo ">>> rendering $page  (fresh GPU execution)"
  rm -rf "content/_freeze/$page" "content/$page/.jupyter_cache" "docs/$page/index_files"
  quarto render "$src"
  echo "<<< done: docs/$page/index.html"
done

echo ""
echo "All requested pages rendered. Next:"
echo "  git add docs/<page>/          # commit the GPU-rendered output"
echo "  git commit -m 'docs: GPU render of <page> on \$(hostname)'"
echo "  git push"
