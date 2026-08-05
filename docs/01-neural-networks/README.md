# Introduction to Neural Networks
Sam Foreman
2025-07-15

<link rel="preconnect" href="https://fonts.googleapis.com">

- [👋 Welcome](#wave-welcome)
- [🗺️ Map of this section](#world_map-map-of-this-section)

## 👋 Welcome

Neural networks are the engine under everything else in this bootcamp:
the LLMs in [\[2\]](../02-llms/index.qmd) and the science models in
[\[3\]](../04-ai-for-science/index.qmd) are all neural networks, just
larger. This section builds them up from the ground: what a single
neuron does, how a network learns from data, and why the whole
enterprise eventually needs a supercomputer.

It picks up where [\[0\] Intro to AI &
HPC](../00-intro-AI-HPC/index.qmd) left off (you should be comfortable
reading Python and a NumPy array) and takes you from a from-scratch
classifier to training that spans many GPUs.

## 🗺️ Map of this section

The pages are meant to be read in order; each builds on the last:

- 📄 [**\[1.0\] Intro to Neural Networks**](./0-intro/index.qmd) — the
  neuron, layers, activations, and how a network learns via gradient
  descent, building directly on the linear-regression example from
  [\[0\]](../00-intro-AI-HPC/6-linear-regression/index.qmd).
- 📗 [**\[1.1\] MNIST Example**](./1-mnist/index.qmd) — train your first
  real classifier on handwritten digits, from a linear model to a
  multi-layer network.
- 📗 [**\[1.2\] Convolutional Networks**](./2-conv-nets/index.qmd) — the
  architecture that made computer vision work, applied to image
  classification.
- 📗 [**\[1.3\] Representation
  Learning**](./3-representation-learning/index.qmd) — how networks
  *learn features*, and a contrastive method that only pays off at scale
  (a preview of *why* we distribute training).
- 📗 [**\[1.4\] Distributed
  Training**](./4-distributed-training/index.qmd) — the bridge to HPC:
  `DDP`, collective communication, and the parallelism strategies that
  let a model train across many GPUs.

Ready? Start with **[\[1.0\] Intro to Neural
Networks](./0-intro/index.qmd)**.
