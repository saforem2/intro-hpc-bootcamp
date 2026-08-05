# Introduction to Large Language Models (LLMs)
Sam Foreman
2025-07-15

<link rel="preconnect" href="https://fonts.googleapis.com">

- [👋 Welcome](#wave-welcome)
- [🗺️ Map of this section](#world_map-map-of-this-section)
- [Overview](#overview)
  - [Topics](#topics)
- [Natural Language Processing (NLP)](#natural-language-processing-nlp)
- [Large Language Models (LLMs)](#large-language-models-llms)
- [References](#references)

> [!NOTE]
>
> ### Authors
>
> Content modified from original content written by Archit Vasan,
> including materials on LLMs by: Varuni Sastri and Carlo Graziani at
> Argonne, and discussion/editorial work by Taylor Childers, Bethany
> Lusch, and Venkat Vishwanath (Argonne)

## 👋 Welcome

This section takes the neural networks from
[\[1\]](../01-neural-networks/index.qmd) and scales them up to
**language models**. You’ll see what makes a Transformer tick, then
train a small GPT from scratch: first in Colab, then with the same
`ezpz` tooling you’ll use at cluster scale in
[\[3\]](../03-advanced-llms/index.qmd). The conceptual pages run on a
laptop; the training examples run anywhere from Colab to a
supercomputer.

## 🗺️ Map of this section

The pages move from concepts → hands-on training → generation &
evaluation:

- 🧠 [**\[2.0\] Intro to LLMs**](0-intro-to-llms/index.qmd) — why
  sequences, tokenization, embeddings, and a guided dissection of
  GPT-2’s internals.
- ⚙️ [**\[2.1\] Parallel Training**](1-parallel-training/index.qmd) — a
  hands-on lab distributing an LLM training run across GPUs.
- 🪶 [**\[2.2\] Shakespeare Example
  (Colab)**](2-shakespeare-example-colab/index.ipynb) — train a
  character-level GPT in the browser; watch gibberish become verse.
- 🎭 [**\[2.3\] Shakespeare from Scratch
  (ezpz)**](3-shakespeare-ezpz/index.qmd) — the same idea rebuilt with
  `ezpz`, ready to scale from a laptop to the cluster.
- 📊 [**\[2.4\] Evaluating LLMs**](4-evaluating-llms/index.qmd) — how to
  tell whether a language model is actually any good.
- 🎲 [**\[2.5\] Decoding &
  Sampling**](5-decoding-and-sampling/index.qmd) — how a model turns
  logits into text: greedy vs sampling, temperature, and top-p, with
  interactive plots you can run.

Ready to scale up? Continue to the 🚀 **[\[3\] Advanced / Large-Scale
LLMs](../03-advanced-llms/index.qmd)** track.

## Overview

Inspiration from the blog posts “The Illustrated Transformer” and “The
Illustrated GPT2” by Jay Alammar, highly recommended reading.

Across the pages in this section you’ll meet the fundamental concepts
behind large language models (LLMs):

### Topics

- Scientific applications for language models
- General overview of Transformers
- Tokenization
- Model Architecture
- Pipeline using HuggingFace
- Model loading

## Natural Language Processing (NLP)

Large Language Models (LLMs) are a subset of Natural Language Processing
(NLP) techniques that focus on understanding and generating human
language. NLP is a field of linguistics / artificial intelligence that
enables computers to interpret, understand, and respond to human
language in a way that is both meaningful and useful.

The following is a list of common NLP tasks, with some examples:

- **Classifying whole sentences**: Getting the sentiment of a review,
  detecting if an email is spam, determining if a sentence is
  grammatically correct or whether two sentences are logically related
  or not.
- **Classifying each word in a sentence**: Identifying the grammatical
  components of a sentence (noun, verb, adjectvie, …), or the named
  entities (person, location, organization, …).
- **Generating Text**: Completing a prompt with auto-generated text,
  filling in the blanks in a text with masked words
- **Extracting an answer from a text**: Given a question and a context,
  extracting the answer to the question based on the information
  provided in the context.
- **Generating a new sentence from an input text**: Translating a text
  into another language, summarizing a text

## Large Language Models (LLMs)

> A large language model (LLM) is an AI model trained on massive amounts
> of text data that can understand and generate human-like text,
> recognize patterns in language, and perform a wide variety of language
> tasks without task-specific training.  
> They represent a significant advancement in the field of natural
> language processing (NLP) (Face 2022).

> [!WARNING]
>
> ### 🚧 Warning
>
> While LLMs are are able to generate (what appears to be) human-like
> text, they are not sentient, and do not have an understanding of the
> world in the way that humans do. They are trained to predict the next
> word in a sentence based on the context of the words that come before
> it, and can generate text that is coherent and relevant to the input
> they receive. However, they do not have a true understanding of the
> meaning of the words they generate, and can sometimes produce text
> that is nonsensical or irrelevant to the input.

Even with the advances in LLMs, many fundamental challenges remain.
These include understanding ambiguity, cultural context, sarcasm and
humor. LLMs address these challenges through massive training on diverse
datasets, but still often fall short of human-level understanding in
many complex scenarios.

## References

I strongly recommend reading:

- [“The Illustrated
  Transformer”](https://jalammar.github.io/illustrated-transformer/) by
  Jay AlammarAlammar also has a useful post dedicated more generally to
  Sequence-to-Sequence modeling
- [LLM Course by 🤗
  HuggingFace](https://huggingface.co/learn/llm-course/chapter1/1)
- [“Visualizing A Neural Machine Translation Model (Mechanics of Seq2seq
  Models With
  Attention)](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/),
  which illustrates the attention mechanism in the context of a more
  generic language translation model.
- [GPT in 60 Lines of
  NumPy](https://jaykmody.com/blog/gpt-from-scratch/)

<div id="refs" class="references csl-bib-body hanging-indent">

<div id="ref-huggingfacecourse" class="csl-entry">

Face, Hugging. 2022. *The Hugging Face Course, 2022*.
<a href="https://huggingface.co/course"
class="uri">Https://huggingface.co/course</a>.

</div>

</div>
