# Introduction to AI on Supercomputers
Sam Foreman
2025-07-15

<link rel="preconnect" href="https://fonts.googleapis.com">

- [👋 Welcome](#wave-welcome)
- [🗺️ Map of this section](#world_map-map-of-this-section)

## 👋 Welcome

This is the on-ramp for the whole bootcamp. Before we train neural
networks [\[1\]](../01-neural-networks/index.qmd) or large language
models [\[2\]](../02-llms/index.qmd), you need two things: a feel for
**what a supercomputer is and how to work on one**, and enough
**hands-on Python + data + ML fundamentals** to follow every example
that comes after.

No prior HPC experience is assumed. The early pages explain the machine
and how to get a Jupyter notebook running on it; the later pages are
runnable lessons: Python, NumPy/pandas, and three worked ML examples you
can execute on a laptop or on the cluster.

## 🗺️ Map of this section

The pages build from “what is a supercomputer” to “train a model on
one”:

- 📄 [**\[0.0\] Compute Systems**](./0-compute-systems/) — what HPC *is*
  and why AI needs it: the trajectory from a single CPU to today’s GPU
  supercomputers.
- 📄 [**\[0.1\] Shared Resources & Scheduling**](./1-shared-resources/)
  — how a cluster is shared: login vs compute nodes, and submitting jobs
  to a scheduler.
- 📄 [**\[0.2\] Jupyter Notebooks**](./2-jupyter-notebooks/) — get an
  interactive notebook running on an HPC login portal.
- 📄 [**\[0.3\] Using Python**](./3-python/) — a fast, runnable Python
  primer aimed at the idioms the rest of the course leans on.
- 📄 [**\[0.4\] Working with Data**](./4-data/) — NumPy arrays, pandas
  DataFrames, and Matplotlib: the tools every ML example uses.
- 📗 [**\[0.5\] MCMC Example**](./5-mcmc-example/) — estimate π with
  Monte Carlo and parallelize it with MPI: your first taste of “many
  processes, one result.”
- 📗 [**\[0.6\] Linear Regression**](./6-linear-regression/) — fit your
  first model with gradient descent; the conceptual seed of all
  neural-network training.
- 📗 [**\[0.7\] Statistical Learning**](./7-statistical-learning/) —
  k-means, classification, and the ML vocabulary the deep-learning
  sections assume.

Ready? Start with **[\[0.0\] Compute Systems](./0-compute-systems/)**.

Link to original slides:
<https://drive.google.com/file/d/1PH6HlXPhsVB1wDcEkfRSZrDQTqWBi7aH/view?usp=sharing>
