# Convolutional Neural Networks
Sam Foreman, Huihuo Zheng, Corey Adams, Bethany Lusch
2025-07-22

<link rel="preconnect" href="https://fonts.googleapis.com">

- [Getting Started](#getting-started)
- [Convolutional Networks: A brief historical
  context](#convolutional-networks-a-brief-historical-context)
- [Convolutional Building Blocks](#convolutional-building-blocks)
  - [Convolutions](#convolutions)
  - [Normalization](#normalization)
  - [Downsampling (And upsampling)](#downsampling-and-upsampling)
  - [More Building Blocks: Padding, Pooling &
    Channels](#more-building-blocks-padding-pooling--channels)
  - [Padding](#padding)
  - [What is the output size ?](#what-is-the-output-size-)
  - [Max Pooling](#max-pooling)
  - [Multiple Channels](#multiple-channels)
  - [Visualizing learned features from
    CNNs](#visualizing-learned-features-from-cnns)
  - [Residual Connections](#residual-connections)
- [Building a ConvNet](#building-a-convnet)
- [Run Training](#run-training)
- [Run Validation](#run-validation)
- [Plot Metrics](#plot-metrics)
  - [Training Metrics](#training-metrics)
  - [Validation Metrics](#validation-metrics)
- [Homework 1](#homework-1)
  - [Training for Multiple Epochs](#training-for-multiple-epochs)
- [Advanced Architectures](#advanced-architectures)
  - [ResNet](#resnet)
  - [U-Nets](#u-nets)
  - [ViTs](#vits)

## Getting Started

[![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/saforem2/intro-hpc-bootcamp/blob/main/content/01-neural-networks/3-conv-nets/index.ipynb)
[![](https://img.shields.io/badge/-View%20on%20GitHub-333333?style=flat&logo=github&labelColor=gray.png)](https://github.com/saforem2/intro-hpc-bootcamp/blob/main/docs/01-neural-networks/3-conv-nets/README.md)

Up until transformers, convolutions were *the* state of the art in
computer vision.

In many ways and applications they still are!

Large Language Models, which are what we’ll focus on the rest of the
series after this lecture, are really good at ordered, \*tokenized data.
But there is lots of data that isn’t *implicitly* ordered like `images`,
and their more general cousins `graphs`.

Today’s lecture focuses on computer vision models, and particularly on
convolutional neural networks. There are a ton of applications you can
do with these, and not nearly enough time to get into them. Check out
the extra references file to see some publications to get you started if
you want to learn more.

Tip: this notebook is much faster on the GPU!

## Convolutional Networks: A brief historical context

Performance on ImageNet over time\[^image-net-historical\]

``` python
%load_ext autoreload
%autoreload 2
import matplotlib_inline.backend_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('retina', 'svg', 'png')
import os
os.environ["TRUECOLOR"] = "1"
import matplotlib as mpl
# mpl.rcParams['figure.dpi'] = 400
```

``` python
import logging

import ambivalent
import ezpz
import matplotlib.pyplot as plt
import seaborn as sns
from rich import print

sns.set_context("notebook")
# sns.set(rc={"figure.dpi": 400, "savefig.dpi": 400})
plt.style.use(ambivalent.STYLES["ambivalent"])
plt.rcParams["figure.figsize"] = [6.4, 4.8]
plt.rcParams["figure.facecolor"] = "none"
```

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">[<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">2025-08-04 12:29:17,617339</span>][<span style="color: #008000; text-decoration-color: #008000">I</span>][<span style="color: #008080; text-decoration-color: #008080; font-style: italic">ezpz</span>/<span style="color: #000080; text-decoration-color: #000080">__init__</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #800080; text-decoration-color: #800080">265</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">ezpz</span>]<span style="color: #c0c0c0; text-decoration-color: #c0c0c0"> </span>Setting logging level to <span style="color: #008000; text-decoration-color: #008000">'INFO'</span> on <span style="color: #008000; text-decoration-color: #008000">'RANK == 0'</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">[<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">2025-08-04 12:29:17,619494</span>][<span style="color: #008000; text-decoration-color: #008000">I</span>][<span style="color: #008080; text-decoration-color: #008080; font-style: italic">ezpz</span>/<span style="color: #000080; text-decoration-color: #000080">__init__</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #800080; text-decoration-color: #800080">266</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">ezpz</span>]<span style="color: #c0c0c0; text-decoration-color: #c0c0c0"> </span>Setting logging level to <span style="color: #008000; text-decoration-color: #008000">'CRITICAL'</span> on all others <span style="color: #008000; text-decoration-color: #008000">'RANK != 0'</span>
</pre>

``` python
# Data
data = {2010: 28, 2011: 26, 2012: 16, 2013: 12, 2014: 7, 2015: 3, 2016: 2.3, 2017: 2.1}
human_error_rate = 5
plt.bar(list(data.keys()), list(data.values()), color="blue")
plt.axhline(y=human_error_rate, color="red", linestyle="--", label="Human error rate")
plt.xlabel("Year")
plt.ylabel("ImageNet Visual Recognition Error Rate (%)")
plt.title("ImageNet Error Rates Over Time")
plt.legend()
plt.show()
```

![](index_files/figure-commonmark/cell-4-output-1.svg)

## Convolutional Building Blocks

``` python
import torch
import torchvision
```

We’re going to go through some examples of building blocks for
convolutional networks. To help illustate some of these, let’s use an
image for examples:

``` python
from PIL import Image

# wget line useful in Google Colab
#! wget https://raw.githubusercontent.com/argonne-lcf/ai-science-training-series/main/03_advanced_neural_networks/ALCF-Staff.jpg
alcf_image = Image.open("ALCF-Staff.jpg")
```

``` python
from matplotlib import pyplot as plt

fx, fy = plt.rcParamsDefault["figure.figsize"]
figure = plt.figure(figsize=(1.5 * fx, 1.5 * fy))
_ = plt.imshow(alcf_image)
```

![](index_files/figure-commonmark/cell-7-output-1.svg)

### Convolutions

$$\begin{equation}
G\left[m, n\right] = \left(f \star h\right)\left[m, n\right] = \sum_{j} \sum_{k} h\left[j, k\right] f\left[m - j, n - k\right]
\end{equation}$$

Convolutions are a restriction of - and a specialization of - dense
linear layers. A convolution of an image produces another image, and
each output pixel is a function of only it’s local neighborhood of
points. This is called an *inductive bias* and is a big reason why
convolutions work for image data: neighboring pixels are correlated and
you can operate on just those pixels at a time.

$G\left[m, n\right] = \left(f \star h\right)\left[m, n\right] = \sum_{j} \sum_{k} h\left[j, k\right] f\left[m - j, n - k\right]$

See examples of convolutions
[here](https://github.com/vdumoulin/conv_arithmetic)

![image.png](./conv.png)

``` python
# Let's apply a convolution to the ALCF Staff photo:
alcf_tensor = torchvision.transforms.ToTensor()(alcf_image)

# Reshape the tensor to have a batch size of 1:
alcf_tensor = alcf_tensor.reshape((1,) + alcf_tensor.shape)

# Create a random convolution:
# shape is: (channels_in, channels_out, kernel_x, kernel_y)
conv_random = torch.rand((3, 3, 15, 15))

alcf_rand = torch.nn.functional.conv2d(alcf_tensor, conv_random)
alcf_rand = (1.0 / alcf_rand.max()) * alcf_rand
print(alcf_rand.shape)
alcf_rand = alcf_rand.reshape(alcf_rand.shape[1:])

print(alcf_tensor.shape)

rand_image = alcf_rand.permute((1, 2, 0)).cpu()
fx, fy = plt.rcParamsDefault["figure.figsize"]
figure = plt.figure(figsize=(1.5 * fx, 1.5 * fy))
_ = plt.imshow(rand_image)
```

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #800080; text-decoration-color: #800080; font-weight: bold">torch.Size</span><span style="font-weight: bold">([</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">3</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1111</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1986</span><span style="font-weight: bold">])</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #800080; text-decoration-color: #800080; font-weight: bold">torch.Size</span><span style="font-weight: bold">([</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">3</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1125</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2000</span><span style="font-weight: bold">])</span>
</pre>

![](index_files/figure-commonmark/cell-8-output-3.svg)

### Normalization

Normalization is the act of transforming the mean and moment of your
data to standard values (usually 0.0 and 1.0). It’s particularly useful
in machine learning since it stabilizes training, and allows higher
learning rates.

<div id="fig-batch-norm-1">

![Batch Norm](./batch_norm.png)

Figure 1: Reference:
[Normalizations](https://arxiv.org/pdf/1903.10520.pdf)

</div>

<div id="fig-batch-norm-2">

![Batch Normalization accelerates training](./batch_norm_effect.png)

Figure 2: Reference: [Batch Norm](https://arxiv.org/pdf/1502.03167.pdf)

</div>

``` python
# Let's apply a normalization to the ALCF Staff photo:
alcf_tensor = torchvision.transforms.ToTensor()(alcf_image)
# Reshape the tensor to have a batch size of 1:
alcf_tensor = alcf_tensor.reshape((1,) + alcf_tensor.shape)
alcf_rand = torch.nn.functional.normalize(alcf_tensor)
alcf_rand = alcf_rand.reshape(alcf_rand.shape[1:])
print(alcf_tensor.shape)
rand_image = alcf_rand.permute((1, 2, 0)).cpu()
fx, fy = plt.rcParamsDefault["figure.figsize"]
figure = plt.figure(figsize=(1.5 * fx, 1.5 * fy))
_ = plt.imshow(rand_image)
```

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #800080; text-decoration-color: #800080; font-weight: bold">torch.Size</span><span style="font-weight: bold">([</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">3</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1125</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2000</span><span style="font-weight: bold">])</span>
</pre>

![](index_files/figure-commonmark/cell-9-output-2.svg)

### Downsampling (And upsampling)

Downsampling is a critical component of convolutional and many vision
models. Because of the local-only nature of convolutional filters,
learning large-range features can be too slow for convergence.
Downsampling of layers can bring information from far away closer,
effectively changing what it means to be “local” as the input to a
convolution.

<div id="fig-downsampling">

![Convolutional Pooling](./conv_pooling.png)

Figure 3:
[Reference](https://www.researchgate.net/publication/333593451_Application_of_Transfer_Learning_Using_Convolutional_Neural_Network_Method_for_Early_Detection_of_Terry's_Nail)

</div>

``` python
# Let's apply a normalization to the ALCF Staff photo:
alcf_tensor = torchvision.transforms.ToTensor()(alcf_image)
# Reshape the tensor to have a batch size of 1:
alcf_tensor = alcf_tensor.reshape((1,) + alcf_tensor.shape)
alcf_rand = torch.nn.functional.max_pool2d(alcf_tensor, 2)
alcf_rand = alcf_rand.reshape(alcf_rand.shape[1:])
print(alcf_tensor.shape)
rand_image = alcf_rand.permute((1, 2, 0)).cpu()
fx, fy = plt.rcParamsDefault["figure.figsize"]
figure = plt.figure(figsize=(1.5 * fx, 1.5 * fy))
_ = plt.imshow(rand_image)
```

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #800080; text-decoration-color: #800080; font-weight: bold">torch.Size</span><span style="font-weight: bold">([</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">3</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1125</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2000</span><span style="font-weight: bold">])</span>
</pre>

![](index_files/figure-commonmark/cell-10-output-2.svg)

### More Building Blocks: Padding, Pooling & Channels

A few more concepts that show up in almost every ConvNet:

### Padding

Adds zeros along the corners of the image to preserve the dimensionality
of the input.

<div id="fig-conv-padding">

![](images/Padding.png)

Figure 4

</div>

### What is the output size ?

$$(\mathrm{N} - \mathrm{F} + 2\mathrm{P}) / \mathrm{S} + 1$$

- where
  - $N$ = dimension of the input image. (ex, for an image of `28x28x1`,
    $N=28$)
  - $F$ = dimension of the filter ($F=3$ for a filter of `3x3`)
  - $S$ = Stride value
  - $P$ = Size of the zero padding used

### Max Pooling

Pooling reduces the dimensionality of the images, with max-pooling being
one of the most widely used.

<div id="fig-conv-max-pool-1">

![](images/MaxpoolSample2.png)

Figure 5

</div>

### Multiple Channels

Usually colored images have multiple channels with RGB values. What
happens to the filter sizes and activation map dimensions in those
cases?

<div id="fig-multiple-channels">

![](images/multiple_channels.png)

Figure 6

</div>

### Visualizing learned features from CNNs

The filters from the initial hidden layers tend to learn low level
features like edges, corners, shapes, colors etc. Filters from the
deeper layers tend to learn high-level features detecting patterns like
wheel, car, etc.

<div id="fig-conv-nets-feature-maps">

![](images/convnets-feature-maps.png)

Figure 7

</div>

### Residual Connections

One issue, quickly encountered when making convolutional networks deeper
and deeper, is the “Vanishing Gradients” problem. As layers were stacked
on top of each other, the size of updates dimished at the earlier layers
of a convolutional network. The paper “Deep Residual Learning for Image
Recognition” solved this by introduction “residual connections” as skip
layers.

Reference: [Deep Residual Learning for Image
Recognition](https://arxiv.org/pdf/1512.03385.pdf)

<div id="fig-residual-layer">

``` mermaid
flowchart TB
    x(("`x`")) --> w1["`weight layer`"]
    w1 --> r1["`ReLU`"]
    r1 --> w2["`weight layer`"]
    w2 --> add((("`+`")))
    x -->|"`identity (skip)`"| add
    add --> r2["`ReLU`"] --> out(("`f(x) + x`"))
classDef block fill:#CCCCCC02,stroke:#838383,stroke-width:1px,color:#838383
classDef red fill:#ff8181,stroke:#333,stroke-width:1px,color:#000
classDef blue fill:#7DCAFF,stroke:#333,stroke-width:1px,color:#000
classDef yellow fill:#FFFF7F,stroke:#333,stroke-width:1px,color:#000
classDef green fill:#98E6A5,stroke:#333,stroke-width:1px,color:#000
classDef purple fill:#FFCBE6,stroke:#333,stroke-width:1px,color:#000
class x red
class w1,w2 blue
class r1,r2 yellow
class add purple
class out green
```

Figure 8: A residual block: the input `x` is added back to the output of
the weight layers via a **skip connection**, so the block learns
`f(x) + x`.

</div>

Compare the performance of the models before and after the introduction
of these layers:

<div id="fig-resnet-performance">

![](./resnet_comparison.png)

Figure 9

</div>

If you have time to read only one paper on computer vision, make it this
one! Resnet was the first model to beat human accuracy on ImageNet and
is one of the most impactful papers in AI ever published.

## Building a ConvNet

In this section we’ll build and apply a conv net to the mnist dataset.
The layers here are loosely based off of the ConvNext architecture. Why?
Because we’re getting into LLM’s soon, and this ConvNet uses LLM
features. ConvNext is an update to the ResNet architecture that
outperforms it.

[ConvNext](https://arxiv.org/abs/2201.03545)

The dataset here is CIFAR-10 - slightly harder than MNIST but still
relatively easy and computationally tractable.

``` python
batch_size = 16
from torchvision import transforms

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))],
)
training_data = torchvision.datasets.CIFAR10(
    root="data",
    download=True,
    train=True,
    transform=transform,
)

test_data = torchvision.datasets.CIFAR10(
    root="data",
    download=True,
    train=False,
    transform=transform,
)

training_data, validation_data = torch.utils.data.random_split(
    training_data, [0.8, 0.2], generator=torch.Generator().manual_seed(55)
)

# The dataloader makes our dataset iterable
train_dataloader = torch.utils.data.DataLoader(
    training_data,
    batch_size=batch_size,
    pin_memory=True,
    shuffle=True,
    num_workers=0,
)
val_dataloader = torch.utils.data.DataLoader(
    validation_data,
    batch_size=batch_size,
    pin_memory=True,
    shuffle=False,
    num_workers=0,
)
classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)
```

``` python
batch, (X, Y) = next(enumerate(train_dataloader))
plt.imshow(X[0].cpu().permute((1, 2, 0)))
plt.show()
```

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">[<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">2025-08-04 12:29:50,397009</span>][<span style="color: #ffff00; text-decoration-color: #ffff00">W</span>][<span style="color: #008080; text-decoration-color: #008080; font-style: italic">matplotlib</span>/<span style="color: #000080; text-decoration-color: #000080">image</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #800080; text-decoration-color: #800080">661</span>]<span style="color: #c0c0c0; text-decoration-color: #c0c0c0"> </span>Clipping input data to the valid range for imshow with RGB data <span style="color: #ff00ff; text-decoration-color: #ff00ff">([</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>..<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">]</span> for floats or <span style="color: #ff00ff; text-decoration-color: #ff00ff">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>..<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">255</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">]</span> for integers<span style="color: #ff00ff; text-decoration-color: #ff00ff">)</span>. Got range <span style="color: #ff00ff; text-decoration-color: #ff00ff">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">-0.9764706</span>..<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.96862745</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">]</span>.
</pre>

![](index_files/figure-commonmark/cell-12-output-2.svg)

``` python
import numpy as np


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
images, labels = next(iter(train_dataloader))

fx, fy = plt.rcParamsDefault["figure.figsize"]
fig = plt.figure(figsize=(2 * fx, 4 * fy))
# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print("\n" + " ".join(f"{classes[labels[j]]:5s}" for j in range(batch_size)))
```

![](index_files/figure-commonmark/cell-13-output-1.svg)

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">
dog   deer  cat   truck car   horse plane dog   car   bird  bird  frog  deer  bird  car   plane
</pre>

This code below is important as our models get bigger: this is wrapping
the pytorch data loaders to put the data onto the GPU!

``` python
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def preprocess(x, y):
    # CIFAR-10 is *color* images so 3 layers!
    x = x.view(-1, 3, 32, 32)
    #  y = y.to(dtype)
    return (x.to(dev), y.to(dev))


class WrappedDataLoader:
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        for b in self.dl:
            yield (self.func(*b))


train_dataloader = WrappedDataLoader(train_dataloader, preprocess)
val_dataloader = WrappedDataLoader(val_dataloader, preprocess)
```

``` python
from typing import Optional

from torch import nn


class Downsampler(nn.Module):
    def __init__(self, in_channels, out_channels, shape, stride=2):
        super(Downsampler, self).__init__()
        self.norm = nn.LayerNorm([in_channels, *shape])
        self.downsample = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=stride,
            stride=stride,
        )

    def forward(self, inputs):
        return self.downsample(self.norm(inputs))


class ConvNextBlock(nn.Module):
    """This block of operations is loosely based on this paper:"""

    def __init__(
        self,
        in_channels,
        shape,
        kernel_size: Optional[None] = None,
    ):
        super(ConvNextBlock, self).__init__()
        # Depthwise, seperable convolution with a large number of output filters:
        kernel_size = [7, 7] if kernel_size is None else kernel_size
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            groups=in_channels,
            kernel_size=kernel_size,
            padding="same",
        )
        self.norm = nn.LayerNorm([in_channels, *shape])
        # Two more convolutions:
        self.conv2 = nn.Conv2d(
            in_channels=in_channels, out_channels=4 * in_channels, kernel_size=1
        )
        self.conv3 = nn.Conv2d(
            in_channels=4 * in_channels, out_channels=in_channels, kernel_size=1
        )

    def forward(self, inputs):
        x = self.conv1(inputs)
        # The normalization layer:
        x = self.norm(x)
        x = self.conv2(x)
        # The non-linear activation layer:
        x = torch.nn.functional.gelu(x)
        x = self.conv3(x)
        # This makes it a residual network:
        return x + inputs


class Classifier(nn.Module):
    def __init__(
        self,
        n_initial_filters,
        n_stages,
        blocks_per_stage,
        kernel_size: Optional[None] = None,
    ):
        super(Classifier, self).__init__()
        # This is a downsampling convolution that will produce patches of output.
        # This is similar to what vision transformers do to tokenize the images.
        self.stem = nn.Conv2d(
            in_channels=3, out_channels=n_initial_filters, kernel_size=1, stride=1
        )
        current_shape = [32, 32]
        self.norm1 = nn.LayerNorm([n_initial_filters, *current_shape])
        # self.norm1 = WrappedLayerNorm()
        current_n_filters = n_initial_filters
        self.layers = nn.Sequential()
        for i, n_blocks in enumerate(range(n_stages)):
            # Add a convnext block series:
            for _ in range(blocks_per_stage):
                self.layers.append(
                    ConvNextBlock(
                        in_channels=current_n_filters,
                        shape=current_shape,
                        kernel_size=kernel_size,
                    )
                )
            # Add a downsampling layer:
            if i != n_stages - 1:
                # Skip downsampling if it's the last layer!
                self.layers.append(
                    Downsampler(
                        in_channels=current_n_filters,
                        out_channels=2 * current_n_filters,
                        shape=current_shape,
                    )
                )
                # Double the number of filters:
                current_n_filters = 2 * current_n_filters
                # Cut the shape in half:
                current_shape = [cs // 2 for cs in current_shape]
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(current_n_filters),
            nn.Linear(current_n_filters, 10),
        )
        # self.norm2 = nn.InstanceNorm2d(current_n_filters)
        # # This brings it down to one channel / class
        # self.bottleneck = nn.Conv2d(in_channels=current_n_filters, out_channels=10,
        #                                   kernel_size=1, stride=1)

    def forward(self, x):
        x = self.stem(x)
        # Apply a normalization after the initial patching:
        x = self.norm1(x)
        # Apply the main chunk of the network:
        x = self.layers(x)
        # Normalize and readout:
        x = nn.functional.avg_pool2d(x, x.shape[2:])
        x = self.head(x)
        return x

        # x = self.norm2(x)
        # x = self.bottleneck(x)

        # # Average pooling of the remaining spatial dimensions (and reshape) makes this label-like:
        # return nn.functional.avg_pool2d(x, kernel_size=x.shape[-2:]).reshape((-1,10))
```

``` python
from torchinfo import summary

model = Classifier(32, 4, 2, kernel_size=(2, 2))
model.to(device=dev)
print(f"\n{summary(model, input_size=(batch_size, 3, 32, 32))}")
```

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">
==========================================================================================
Layer <span style="font-weight: bold">(</span>typ<span style="color: #00ff00; text-decoration-color: #00ff00; font-weight: bold">e:de</span>pth-idx<span style="font-weight: bold">)</span>                   Output Shape              Param #
==========================================================================================
Classifier                               <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">16</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">10</span><span style="font-weight: bold">]</span>                  --
├─Conv2d: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>-<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>                            <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">16</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">32</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">32</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">32</span><span style="font-weight: bold">]</span>          <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">128</span>
├─LayerNorm: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>-<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span>                         <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">16</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">32</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">32</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">32</span><span style="font-weight: bold">]</span>          <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">65</span>,<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">536</span>
├─Sequential: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>-<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">3</span>                        <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">16</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">256</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">4</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">4</span><span style="font-weight: bold">]</span>           --
│    └─ConvNextBlock: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span>-<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>                <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">16</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">32</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">32</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">32</span><span style="font-weight: bold">]</span>          --
│    │    └─Conv2d: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">3</span>-<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>                  <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">16</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">32</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">32</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">32</span><span style="font-weight: bold">]</span>          <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">160</span>
│    │    └─LayerNorm: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">3</span>-<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span>               <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">16</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">32</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">32</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">32</span><span style="font-weight: bold">]</span>          <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">65</span>,<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">536</span>
│    │    └─Conv2d: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">3</span>-<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">3</span>                  <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">16</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">128</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">32</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">32</span><span style="font-weight: bold">]</span>         <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">4</span>,<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">224</span>
│    │    └─Conv2d: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">3</span>-<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">4</span>                  <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">16</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">32</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">32</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">32</span><span style="font-weight: bold">]</span>          <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">4</span>,<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">128</span>
│    └─ConvNextBlock: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span>-<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span>                <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">16</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">32</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">32</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">32</span><span style="font-weight: bold">]</span>          --
│    │    └─Conv2d: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">3</span>-<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">5</span>                  <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">16</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">32</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">32</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">32</span><span style="font-weight: bold">]</span>          <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">160</span>
│    │    └─LayerNorm: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">3</span>-<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">6</span>               <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">16</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">32</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">32</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">32</span><span style="font-weight: bold">]</span>          <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">65</span>,<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">536</span>
│    │    └─Conv2d: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">3</span>-<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">7</span>                  <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">16</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">128</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">32</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">32</span><span style="font-weight: bold">]</span>         <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">4</span>,<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">224</span>
│    │    └─Conv2d: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">3</span>-<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">8</span>                  <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">16</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">32</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">32</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">32</span><span style="font-weight: bold">]</span>          <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">4</span>,<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">128</span>
│    └─Downsampler: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span>-<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">3</span>                  <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">16</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">64</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">16</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">16</span><span style="font-weight: bold">]</span>          --
│    │    └─LayerNorm: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">3</span>-<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">9</span>               <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">16</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">32</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">32</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">32</span><span style="font-weight: bold">]</span>          <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">65</span>,<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">536</span>
│    │    └─Conv2d: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">3</span>-<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">10</span>                 <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">16</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">64</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">16</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">16</span><span style="font-weight: bold">]</span>          <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">8</span>,<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">256</span>
│    └─ConvNextBlock: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span>-<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">4</span>                <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">16</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">64</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">16</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">16</span><span style="font-weight: bold">]</span>          --
│    │    └─Conv2d: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">3</span>-<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">11</span>                 <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">16</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">64</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">16</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">16</span><span style="font-weight: bold">]</span>          <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">320</span>
│    │    └─LayerNorm: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">3</span>-<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">12</span>              <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">16</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">64</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">16</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">16</span><span style="font-weight: bold">]</span>          <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">32</span>,<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">768</span>
│    │    └─Conv2d: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">3</span>-<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">13</span>                 <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">16</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">256</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">16</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">16</span><span style="font-weight: bold">]</span>         <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">16</span>,<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">640</span>
│    │    └─Conv2d: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">3</span>-<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">14</span>                 <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">16</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">64</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">16</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">16</span><span style="font-weight: bold">]</span>          <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">16</span>,<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">448</span>
│    └─ConvNextBlock: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span>-<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">5</span>                <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">16</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">64</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">16</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">16</span><span style="font-weight: bold">]</span>          --
│    │    └─Conv2d: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">3</span>-<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">15</span>                 <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">16</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">64</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">16</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">16</span><span style="font-weight: bold">]</span>          <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">320</span>
│    │    └─LayerNorm: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">3</span>-<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">16</span>              <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">16</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">64</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">16</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">16</span><span style="font-weight: bold">]</span>          <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">32</span>,<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">768</span>
│    │    └─Conv2d: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">3</span>-<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">17</span>                 <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">16</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">256</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">16</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">16</span><span style="font-weight: bold">]</span>         <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">16</span>,<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">640</span>
│    │    └─Conv2d: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">3</span>-<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">18</span>                 <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">16</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">64</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">16</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">16</span><span style="font-weight: bold">]</span>          <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">16</span>,<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">448</span>
│    └─Downsampler: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span>-<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">6</span>                  <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">16</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">128</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">8</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">8</span><span style="font-weight: bold">]</span>           --
│    │    └─LayerNorm: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">3</span>-<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">19</span>              <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">16</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">64</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">16</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">16</span><span style="font-weight: bold">]</span>          <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">32</span>,<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">768</span>
│    │    └─Conv2d: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">3</span>-<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">20</span>                 <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">16</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">128</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">8</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">8</span><span style="font-weight: bold">]</span>           <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">32</span>,<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">896</span>
│    └─ConvNextBlock: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span>-<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">7</span>                <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">16</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">128</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">8</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">8</span><span style="font-weight: bold">]</span>           --
│    │    └─Conv2d: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">3</span>-<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">21</span>                 <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">16</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">128</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">8</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">8</span><span style="font-weight: bold">]</span>           <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">640</span>
│    │    └─LayerNorm: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">3</span>-<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">22</span>              <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">16</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">128</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">8</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">8</span><span style="font-weight: bold">]</span>           <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">16</span>,<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">384</span>
│    │    └─Conv2d: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">3</span>-<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">23</span>                 <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">16</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">512</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">8</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">8</span><span style="font-weight: bold">]</span>           <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">66</span>,<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">048</span>
│    │    └─Conv2d: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">3</span>-<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">24</span>                 <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">16</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">128</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">8</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">8</span><span style="font-weight: bold">]</span>           <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">65</span>,<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">664</span>
│    └─ConvNextBlock: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span>-<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">8</span>                <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">16</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">128</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">8</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">8</span><span style="font-weight: bold">]</span>           --
│    │    └─Conv2d: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">3</span>-<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">25</span>                 <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">16</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">128</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">8</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">8</span><span style="font-weight: bold">]</span>           <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">640</span>
│    │    └─LayerNorm: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">3</span>-<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">26</span>              <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">16</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">128</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">8</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">8</span><span style="font-weight: bold">]</span>           <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">16</span>,<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">384</span>
│    │    └─Conv2d: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">3</span>-<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">27</span>                 <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">16</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">512</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">8</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">8</span><span style="font-weight: bold">]</span>           <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">66</span>,<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">048</span>
│    │    └─Conv2d: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">3</span>-<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">28</span>                 <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">16</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">128</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">8</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">8</span><span style="font-weight: bold">]</span>           <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">65</span>,<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">664</span>
│    └─Downsampler: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span>-<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">9</span>                  <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">16</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">256</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">4</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">4</span><span style="font-weight: bold">]</span>           --
│    │    └─LayerNorm: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">3</span>-<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">29</span>              <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">16</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">128</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">8</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">8</span><span style="font-weight: bold">]</span>           <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">16</span>,<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">384</span>
│    │    └─Conv2d: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">3</span>-<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">30</span>                 <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">16</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">256</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">4</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">4</span><span style="font-weight: bold">]</span>           <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">131</span>,<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">328</span>
│    └─ConvNextBlock: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span>-<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">10</span>               <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">16</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">256</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">4</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">4</span><span style="font-weight: bold">]</span>           --
│    │    └─Conv2d: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">3</span>-<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">31</span>                 <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">16</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">256</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">4</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">4</span><span style="font-weight: bold">]</span>           <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>,<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">280</span>
│    │    └─LayerNorm: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">3</span>-<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">32</span>              <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">16</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">256</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">4</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">4</span><span style="font-weight: bold">]</span>           <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">8</span>,<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">192</span>
│    │    └─Conv2d: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">3</span>-<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">33</span>                 <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">16</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1024</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">4</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">4</span><span style="font-weight: bold">]</span>          <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">263</span>,<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">168</span>
│    │    └─Conv2d: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">3</span>-<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">34</span>                 <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">16</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">256</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">4</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">4</span><span style="font-weight: bold">]</span>           <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">262</span>,<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">400</span>
│    └─ConvNextBlock: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span>-<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">11</span>               <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">16</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">256</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">4</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">4</span><span style="font-weight: bold">]</span>           --
│    │    └─Conv2d: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">3</span>-<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">35</span>                 <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">16</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">256</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">4</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">4</span><span style="font-weight: bold">]</span>           <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>,<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">280</span>
│    │    └─LayerNorm: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">3</span>-<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">36</span>              <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">16</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">256</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">4</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">4</span><span style="font-weight: bold">]</span>           <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">8</span>,<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">192</span>
│    │    └─Conv2d: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">3</span>-<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">37</span>                 <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">16</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1024</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">4</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">4</span><span style="font-weight: bold">]</span>          <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">263</span>,<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">168</span>
│    │    └─Conv2d: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">3</span>-<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">38</span>                 <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">16</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">256</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">4</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">4</span><span style="font-weight: bold">]</span>           <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">262</span>,<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">400</span>
├─Sequential: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>-<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">4</span>                        <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">16</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">10</span><span style="font-weight: bold">]</span>                  --
│    └─Flatten: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span>-<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">12</span>                     <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">16</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">256</span><span style="font-weight: bold">]</span>                 --
│    └─LayerNorm: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span>-<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">13</span>                   <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">16</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">256</span><span style="font-weight: bold">]</span>                 <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">512</span>
│    └─Linear: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span>-<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">14</span>                      <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">16</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">10</span><span style="font-weight: bold">]</span>                  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span>,<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">570</span>
==========================================================================================
Total params: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span>,<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">003</span>,<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">914</span>
Trainable params: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span>,<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">003</span>,<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">914</span>
Non-trainable params: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>
Total mult-adds <span style="font-weight: bold">(</span>Units.GIGABYTES<span style="font-weight: bold">)</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1.20</span>
==========================================================================================
Input size <span style="font-weight: bold">(</span>MB<span style="font-weight: bold">)</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.20</span>
Forward/backward pass size <span style="font-weight: bold">(</span>MB<span style="font-weight: bold">)</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">129.53</span>
Params size <span style="font-weight: bold">(</span>MB<span style="font-weight: bold">)</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">8.02</span>
Estimated Total Size <span style="font-weight: bold">(</span>MB<span style="font-weight: bold">)</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">137.75</span>
==========================================================================================
</pre>

``` python
def evaluate(dataloader, model, loss_fn, val_bar):
    # Set the model to evaluation mode - some NN pieces behave differently during training
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader)
    num_batches = len(dataloader)
    loss, correct = 0, 0

    # We can save computation and memory by not calculating gradients here - we aren't optimizing
    with torch.no_grad():
        # loop over all of the batches
        for x, y in dataloader:
            t0 = time.perf_counter()
            pred = model(x.to(DTYPE))
            t1 = time.perf_counter()
            loss += loss_fn(pred, y).item()
            # how many are correct in this batch? Tracking for accuracy
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            t3 = time.perf_counter()
            val_bar.update()

    loss /= num_batches
    correct /= size * batch_size

    accuracy = 100 * correct
    return accuracy, loss
```

``` python
import time

from torch import nn

DTYPE = torch.bfloat16
DEVICE = ezpz.get_torch_device_type()

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=2.5e-4)
```

``` python
def eval_step(x, y):
    with torch.no_grad():
        t0 = time.perf_counter()
        pred = model(x.to(DTYPE))
        t1 = time.perf_counter()
        loss = loss_fn(pred, y).item()
        correct = (pred.argmax(1) == y).type(torch.float).sum().item()
        t2 = time.perf_counter()
    return {
        "loss": loss,
        "acc": correct / y.shape[0],
        "dtf": t1 - t0,
        "dtm": t2 - t1,
    }
```

``` python
def train_step(x, y):
    t0 = time.perf_counter()
    # Forward pass
    with torch.autocast(dtype=DTYPE, device_type=DEVICE):
        pred = model(x.to(DTYPE))
    loss = loss_fn(pred, y)
    t1 = time.perf_counter()

    # Backward pass
    loss.backward()
    t2 = time.perf_counter()

    # Update weights
    optimizer.step()
    t3 = time.perf_counter()

    # Reset gradients
    optimizer.zero_grad()
    t4 = time.perf_counter()

    return loss.item(), {
        "dtf": t1 - t0,
        "dtb": t2 - t1,
        "dtu": t3 - t2,
        "dtz": t4 - t3,
    }
```

``` python
def train_one_epoch(
    dataloader, model, loss_fn, optimizer, progress_bar, history: ezpz.History | None
):
    model.train()
    t0 = time.perf_counter()
    batch_metrics = {}
    for batch, (X, y) in enumerate(dataloader):
        loss, metrics = train_step(x, y)
        progress_bar.update()
        metrics = {"bidx": batch, "loss": loss, **metrics}
        batch_metrics[batch] = metrics
        if history is not None:
            print(history.update(metrics))
    t1 = time.perf_counter()
    batch_metrics |= {"dt_batch": t1 - t0}
    # if history is not None:
    #     _ = history.update({"dt_batch": t1 - t0})
    return batch_metrics
```

``` python
def train_one_epoch1(
    dataloader, model, loss_fn, optimizer, progress_bar, history: ezpz.History | None
):
    model.train()
    t0 = time.perf_counter()
    batch_metrics = {}
    for batch, (X, y) in enumerate(dataloader):
        _t0 = time.perf_counter()
        # forward pass
        pred = model(X)
        _t1 = time.perf_counter()
        loss = loss_fn(pred, y)
        _t2 = time.perf_counter()
        # backward pass calculates gradients
        loss.backward()
        _t3 = time.perf_counter()
        # take one step with these gradients
        optimizer.step()
        _t4 = time.perf_counter()
        # resets the gradients
        optimizer.zero_grad()
        _t5 = time.perf_counter()
        progress_bar.update()
        metrics = {
            "bidx": batch,
            "loss": loss.item(),
            "dtf": (_t1 - _t0),
            "dtl": (_t2 - _t1),
            "dtb": (_t3 - _t2),
            "dto": (_t4 - _t3),
            "dtz": (_t5 - _t4),
        }
        batch_metrics[batch] = metrics
        if history is not None:
            summary = history.update(metrics)
    t1 = time.perf_counter()
    batch_metrics |= {
        "dt_batch": t1 - t0,
    }
    return batch_metrics
```

``` python
_ = model.to(DTYPE)
```

``` python
_x, _y = next(iter(val_dataloader))
print(f"{eval_step(_x.to(DTYPE), _y)}")
```

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">{</span><span style="color: #008000; text-decoration-color: #008000">'loss'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2.40625</span>, <span style="color: #008000; text-decoration-color: #008000">'acc'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.125</span>, <span style="color: #008000; text-decoration-color: #008000">'dtf'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.1633995003066957</span>, <span style="color: #008000; text-decoration-color: #008000">'dtm'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.0010725418105721474</span><span style="font-weight: bold">}</span>
</pre>

``` python
print(f"{_x.shape=}, {_y.shape=}")
_pred = model(_x.to(DTYPE))
_loss = loss_fn(_pred, _y).item()
_correct = (_pred.argmax(1) == _y).type(torch.float).sum().item()
print(
    {
        # "pred": _pred,
        "loss": _loss,
        "pred.argmax(1)": _pred.argmax(1),
        "pred.argmax(1) == y": (_pred.argmax(1) == _y),
        "correct": _correct,
        "acc": _correct / _y.shape[0],
    }
)
```

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">_x.<span style="color: #808000; text-decoration-color: #808000">shape</span>=<span style="color: #800080; text-decoration-color: #800080; font-weight: bold">torch</span><span style="color: #800080; text-decoration-color: #800080; font-weight: bold">.Size</span><span style="font-weight: bold">([</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">16</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">3</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">32</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">32</span><span style="font-weight: bold">])</span>, _y.<span style="color: #808000; text-decoration-color: #808000">shape</span>=<span style="color: #800080; text-decoration-color: #800080; font-weight: bold">torch</span><span style="color: #800080; text-decoration-color: #800080; font-weight: bold">.Size</span><span style="font-weight: bold">([</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">16</span><span style="font-weight: bold">])</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">{</span>
    <span style="color: #008000; text-decoration-color: #008000">'loss'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2.40625</span>,
    <span style="color: #008000; text-decoration-color: #008000">'pred.argmax(1)'</span>: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">tensor</span><span style="font-weight: bold">([</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">4</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">4</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">4</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">4</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span><span style="font-weight: bold">])</span>,
    <span style="color: #008000; text-decoration-color: #008000">'pred.argmax(1) == y'</span>: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">tensor</span><span style="font-weight: bold">([</span><span style="color: #ff0000; text-decoration-color: #ff0000; font-style: italic">False</span>, <span style="color: #ff0000; text-decoration-color: #ff0000; font-style: italic">False</span>, <span style="color: #ff0000; text-decoration-color: #ff0000; font-style: italic">False</span>, <span style="color: #ff0000; text-decoration-color: #ff0000; font-style: italic">False</span>,  <span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">True</span>, <span style="color: #ff0000; text-decoration-color: #ff0000; font-style: italic">False</span>,  <span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">True</span>, <span style="color: #ff0000; text-decoration-color: #ff0000; font-style: italic">False</span>, <span style="color: #ff0000; text-decoration-color: #ff0000; font-style: italic">False</span>, <span style="color: #ff0000; text-decoration-color: #ff0000; font-style: italic">False</span>,
        <span style="color: #ff0000; text-decoration-color: #ff0000; font-style: italic">False</span>, <span style="color: #ff0000; text-decoration-color: #ff0000; font-style: italic">False</span>, <span style="color: #ff0000; text-decoration-color: #ff0000; font-style: italic">False</span>, <span style="color: #ff0000; text-decoration-color: #ff0000; font-style: italic">False</span>, <span style="color: #ff0000; text-decoration-color: #ff0000; font-style: italic">False</span>, <span style="color: #ff0000; text-decoration-color: #ff0000; font-style: italic">False</span><span style="font-weight: bold">])</span>,
    <span style="color: #008000; text-decoration-color: #008000">'correct'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2.0</span>,
    <span style="color: #008000; text-decoration-color: #008000">'acc'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.125</span>
<span style="font-weight: bold">}</span>
</pre>

## Run Training

``` python
import time

import ezpz
from tqdm.auto import tqdm, trange

PRINT_EVERY = 50
TRAIN_ITERS = 500

history = ezpz.History()
model.train()
for i in trange(TRAIN_ITERS, desc="Training"):
    t0 = time.perf_counter()
    x, y = next(iter(train_dataloader))
    t1 = time.perf_counter()
    loss, dt = train_step(x, y)
    summary = history.update(
        {
            "train/iter": i,
            "train/loss": loss,
            "train/dtd": t1 - t0,
            **{f"train/{k}": v for k, v in dt.items()},
        },
    ).replace("/", ".")
    if i % PRINT_EVERY == 0:
        print(summary)
```

    Training:   0%|          | 0/500 [00:00<?, ?it/s]

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">train.<span style="color: #808000; text-decoration-color: #808000">iter</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span> train.<span style="color: #808000; text-decoration-color: #808000">loss</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2.515625</span> train.<span style="color: #808000; text-decoration-color: #808000">dtd</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.010065</span> train.<span style="color: #808000; text-decoration-color: #808000">dtf</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.171240</span> train.<span style="color: #808000; text-decoration-color: #808000">dtb</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.713570</span> train.<span style="color: #808000; text-decoration-color: #808000">dtu</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.013012</span> 
train.<span style="color: #808000; text-decoration-color: #808000">dtz</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.000208</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">train.<span style="color: #808000; text-decoration-color: #808000">iter</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">50</span> train.<span style="color: #808000; text-decoration-color: #808000">loss</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1.953125</span> train.<span style="color: #808000; text-decoration-color: #808000">dtd</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.002513</span> train.<span style="color: #808000; text-decoration-color: #808000">dtf</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.146031</span> train.<span style="color: #808000; text-decoration-color: #808000">dtb</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.657839</span> train.<span style="color: #808000; text-decoration-color: #808000">dtu</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.008844</span> 
train.<span style="color: #808000; text-decoration-color: #808000">dtz</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.000213</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">train.<span style="color: #808000; text-decoration-color: #808000">iter</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">100</span> train.<span style="color: #808000; text-decoration-color: #808000">loss</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2.031250</span> train.<span style="color: #808000; text-decoration-color: #808000">dtd</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.002231</span> train.<span style="color: #808000; text-decoration-color: #808000">dtf</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.140829</span> train.<span style="color: #808000; text-decoration-color: #808000">dtb</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.657641</span> train.<span style="color: #808000; text-decoration-color: #808000">dtu</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.008523</span> 
train.<span style="color: #808000; text-decoration-color: #808000">dtz</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.000480</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">train.<span style="color: #808000; text-decoration-color: #808000">iter</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">150</span> train.<span style="color: #808000; text-decoration-color: #808000">loss</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2.125000</span> train.<span style="color: #808000; text-decoration-color: #808000">dtd</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.001999</span> train.<span style="color: #808000; text-decoration-color: #808000">dtf</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.145267</span> train.<span style="color: #808000; text-decoration-color: #808000">dtb</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.679628</span> train.<span style="color: #808000; text-decoration-color: #808000">dtu</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.008671</span> 
train.<span style="color: #808000; text-decoration-color: #808000">dtz</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.000192</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">train.<span style="color: #808000; text-decoration-color: #808000">iter</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">200</span> train.<span style="color: #808000; text-decoration-color: #808000">loss</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1.890625</span> train.<span style="color: #808000; text-decoration-color: #808000">dtd</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.002067</span> train.<span style="color: #808000; text-decoration-color: #808000">dtf</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.141468</span> train.<span style="color: #808000; text-decoration-color: #808000">dtb</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.656727</span> train.<span style="color: #808000; text-decoration-color: #808000">dtu</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.008663</span> 
train.<span style="color: #808000; text-decoration-color: #808000">dtz</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.000650</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">train.<span style="color: #808000; text-decoration-color: #808000">iter</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">250</span> train.<span style="color: #808000; text-decoration-color: #808000">loss</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1.984375</span> train.<span style="color: #808000; text-decoration-color: #808000">dtd</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.002972</span> train.<span style="color: #808000; text-decoration-color: #808000">dtf</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.138405</span> train.<span style="color: #808000; text-decoration-color: #808000">dtb</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.662314</span> train.<span style="color: #808000; text-decoration-color: #808000">dtu</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.008473</span> 
train.<span style="color: #808000; text-decoration-color: #808000">dtz</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.000427</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">train.<span style="color: #808000; text-decoration-color: #808000">iter</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">300</span> train.<span style="color: #808000; text-decoration-color: #808000">loss</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2.281250</span> train.<span style="color: #808000; text-decoration-color: #808000">dtd</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.001965</span> train.<span style="color: #808000; text-decoration-color: #808000">dtf</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.149045</span> train.<span style="color: #808000; text-decoration-color: #808000">dtb</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.649204</span> train.<span style="color: #808000; text-decoration-color: #808000">dtu</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.008485</span> 
train.<span style="color: #808000; text-decoration-color: #808000">dtz</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.000199</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">train.<span style="color: #808000; text-decoration-color: #808000">iter</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">350</span> train.<span style="color: #808000; text-decoration-color: #808000">loss</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1.851562</span> train.<span style="color: #808000; text-decoration-color: #808000">dtd</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.002693</span> train.<span style="color: #808000; text-decoration-color: #808000">dtf</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.149543</span> train.<span style="color: #808000; text-decoration-color: #808000">dtb</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.677236</span> train.<span style="color: #808000; text-decoration-color: #808000">dtu</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.008860</span> 
train.<span style="color: #808000; text-decoration-color: #808000">dtz</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.000135</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">train.<span style="color: #808000; text-decoration-color: #808000">iter</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">400</span> train.<span style="color: #808000; text-decoration-color: #808000">loss</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1.812500</span> train.<span style="color: #808000; text-decoration-color: #808000">dtd</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.002285</span> train.<span style="color: #808000; text-decoration-color: #808000">dtf</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.136938</span> train.<span style="color: #808000; text-decoration-color: #808000">dtb</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.683250</span> train.<span style="color: #808000; text-decoration-color: #808000">dtu</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.008446</span> 
train.<span style="color: #808000; text-decoration-color: #808000">dtz</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.000188</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">train.<span style="color: #808000; text-decoration-color: #808000">iter</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">450</span> train.<span style="color: #808000; text-decoration-color: #808000">loss</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2.265625</span> train.<span style="color: #808000; text-decoration-color: #808000">dtd</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.002329</span> train.<span style="color: #808000; text-decoration-color: #808000">dtf</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.149435</span> train.<span style="color: #808000; text-decoration-color: #808000">dtb</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.653462</span> train.<span style="color: #808000; text-decoration-color: #808000">dtu</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.008714</span> 
train.<span style="color: #808000; text-decoration-color: #808000">dtz</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.000223</span>
</pre>

## Run Validation

``` python
eval_history = ezpz.History()
model.eval()
PRINT_EVERY = 50
# EVAL_ITERS = 50

with torch.no_grad():
    for bidx, (x, y) in enumerate(tqdm(val_dataloader)):
        t0 = time.perf_counter()
        pred = model(x.to(DTYPE))
        loss = loss_fn(pred, y).item()
        correct = (pred.argmax(1) == y).to(torch.float).sum().item()
        acc = correct / y.shape[0]
        metrics = {
            "val/iter": bidx,
            "val/loss": loss,
            "val/acc": acc,
        }
        summary = eval_history.update(metrics)
        if bidx % PRINT_EVERY == 0:
            print(summary)
```

      0%|          | 0/625 [00:00<?, ?it/s]

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">val/<span style="color: #808000; text-decoration-color: #808000">iter</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span> val/<span style="color: #808000; text-decoration-color: #808000">loss</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1.734375</span> val/<span style="color: #808000; text-decoration-color: #808000">acc</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.312500</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">val/<span style="color: #808000; text-decoration-color: #808000">iter</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">50</span> val/<span style="color: #808000; text-decoration-color: #808000">loss</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1.664062</span> val/<span style="color: #808000; text-decoration-color: #808000">acc</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.437500</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">val/<span style="color: #808000; text-decoration-color: #808000">iter</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">100</span> val/<span style="color: #808000; text-decoration-color: #808000">loss</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1.882812</span> val/<span style="color: #808000; text-decoration-color: #808000">acc</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.312500</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">val/<span style="color: #808000; text-decoration-color: #808000">iter</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">150</span> val/<span style="color: #808000; text-decoration-color: #808000">loss</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1.609375</span> val/<span style="color: #808000; text-decoration-color: #808000">acc</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.562500</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">val/<span style="color: #808000; text-decoration-color: #808000">iter</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">200</span> val/<span style="color: #808000; text-decoration-color: #808000">loss</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1.921875</span> val/<span style="color: #808000; text-decoration-color: #808000">acc</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.250000</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">val/<span style="color: #808000; text-decoration-color: #808000">iter</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">250</span> val/<span style="color: #808000; text-decoration-color: #808000">loss</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1.609375</span> val/<span style="color: #808000; text-decoration-color: #808000">acc</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.437500</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">val/<span style="color: #808000; text-decoration-color: #808000">iter</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">300</span> val/<span style="color: #808000; text-decoration-color: #808000">loss</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1.484375</span> val/<span style="color: #808000; text-decoration-color: #808000">acc</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.562500</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">val/<span style="color: #808000; text-decoration-color: #808000">iter</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">350</span> val/<span style="color: #808000; text-decoration-color: #808000">loss</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1.617188</span> val/<span style="color: #808000; text-decoration-color: #808000">acc</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.437500</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">val/<span style="color: #808000; text-decoration-color: #808000">iter</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">400</span> val/<span style="color: #808000; text-decoration-color: #808000">loss</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2.218750</span> val/<span style="color: #808000; text-decoration-color: #808000">acc</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.125000</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">val/<span style="color: #808000; text-decoration-color: #808000">iter</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">450</span> val/<span style="color: #808000; text-decoration-color: #808000">loss</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1.484375</span> val/<span style="color: #808000; text-decoration-color: #808000">acc</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.437500</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">val/<span style="color: #808000; text-decoration-color: #808000">iter</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">500</span> val/<span style="color: #808000; text-decoration-color: #808000">loss</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1.781250</span> val/<span style="color: #808000; text-decoration-color: #808000">acc</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.250000</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">val/<span style="color: #808000; text-decoration-color: #808000">iter</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">550</span> val/<span style="color: #808000; text-decoration-color: #808000">loss</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1.687500</span> val/<span style="color: #808000; text-decoration-color: #808000">acc</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.375000</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">val/<span style="color: #808000; text-decoration-color: #808000">iter</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">600</span> val/<span style="color: #808000; text-decoration-color: #808000">loss</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1.609375</span> val/<span style="color: #808000; text-decoration-color: #808000">acc</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.437500</span>
</pre>

## Plot Metrics

### Training Metrics

``` python
ezpz.plot.plot_dataset((tdset := history.get_dataset()), save_plots=False)
```

![](index_files/figure-commonmark/cell-28-output-1.svg)

![](index_files/figure-commonmark/cell-28-output-2.svg)

![](index_files/figure-commonmark/cell-28-output-3.svg)

![](index_files/figure-commonmark/cell-28-output-4.svg)

![](index_files/figure-commonmark/cell-28-output-5.svg)

![](index_files/figure-commonmark/cell-28-output-6.svg)

![](index_files/figure-commonmark/cell-28-output-7.svg)

![](index_files/figure-commonmark/cell-28-output-8.svg)

### Validation Metrics

``` python
ezpz.plot.plot_dataset((edset := eval_history.get_dataset()), save_plots=False)
```

![](index_files/figure-commonmark/cell-29-output-1.svg)

![](index_files/figure-commonmark/cell-29-output-2.svg)

![](index_files/figure-commonmark/cell-29-output-3.svg)

![](index_files/figure-commonmark/cell-29-output-4.svg)

------------------------------------------------------------------------

## Homework 1

In this notebook, we’ve learned about some basic convolutional networks
and trained one on CIFAR-10 images. It did … OK. There is significant
overfitting of this model. There are some ways to address that, but we
didn’t have time to get into that in this session.

Meanwhile, your homework (part 1) for this week is to try to train the
model again but with a different architecture. Change one or more of the
following: - The number of convolutions between downsampling - The
number of filters in each layer - The initial “patchify” layer - Another
hyper-parameter of your choosing

And compare your final validation accuracy to the accuracy shown here.
Can you beat the validation accuracy shown?

For full credit on the homework, you need to show (via text, or make a
plot) the training and validation data sets’ performance (loss and
accuracy) for all the epochs you train. You also need to explain, in
several sentences, what you changed in the network and why you think it
makes a difference.

### Training for Multiple Epochs

``` python
epochs = 1
train_history = ezpz.History()
for j in range(epochs):
    with tqdm(
        total=len(train_dataloader), position=0, leave=True, desc=f"Train Epoch {j}"
    ) as train_bar:
        bmetrics = train_one_epoch(
            train_dataloader,
            model,
            loss_fn,
            optimizer,
            train_bar,
            history=train_history,
        )

    # checking on the training & validation loss & accuracy
    # for training data - only once every 5 epochs (takes a while)
    if j % 5 == 0:
        with tqdm(
            total=len(train_dataloader),
            position=0,
            leave=True,
            desc=f"Validate (train) Epoch {j}",
        ) as train_eval:
            acc, loss = evaluate(train_dataloader, model, loss_fn, train_eval)
            print(f"Epoch {j}: training loss: {loss:.3f}, accuracy: {acc:.3f}")

    with tqdm(
        total=len(val_dataloader), position=0, leave=True, desc=f"Validate Epoch {j}"
    ) as val_bar:
        acc_val, loss_val = evaluate(val_dataloader, model, loss_fn, val_bar)
        print(
            f"Epoch {j}: validation loss: {loss_val:.3f}, accuracy: {acc_val:.3f}"
        )
```

## Advanced Architectures

The ConvNet you built above is the foundation. Real-world models stack
these building blocks into deeper, more specialized architectures. Here
are three influential ones.

### ResNet

Deeper and deeper networks that stack convolutions end up with smaller
and smaller gradients in early layers. ResNets use additional skip
connections where the output layer is `f(x) + x` instead of `f(w x + b)`
or `f(x)`. This avoids vanishing gradient problem and results in
smoother loss functions. Refer to the [ResNet
paper](https://arxiv.org/pdf/1512.03385.pdf) and [ResNet loss
visualization paper](https://arxiv.org/pdf/1712.09913.pdf) for more
information.

<div id="fig-resnet">

``` mermaid
flowchart LR
    i(("`input`")) --> b1
    subgraph b1["`Res block 1`"]
        c1["`conv`"] --> a1((("`+`")))
    end
    subgraph b2["`Res block 2`"]
        c2["`conv`"] --> a2((("`+`")))
    end
    subgraph b3["`Res block 3`"]
        c3["`conv`"] --> a3((("`+`")))
    end
    b1 --> b2 --> b3 --> o(("`output`"))
    i -.->|"`skip`"| a1
    b1 -.->|"`skip`"| a2
    b2 -.->|"`skip`"| a3
classDef block fill:#CCCCCC02,stroke:#838383,stroke-width:1px,color:#838383
classDef red fill:#ff8181,stroke:#333,stroke-width:1px,color:#000
classDef blue fill:#7DCAFF,stroke:#333,stroke-width:1px,color:#000
classDef yellow fill:#FFFF7F,stroke:#333,stroke-width:1px,color:#000
classDef green fill:#98E6A5,stroke:#333,stroke-width:1px,color:#000
classDef purple fill:#FFCBE6,stroke:#333,stroke-width:1px,color:#000
class b1,b2,b3 block
class i red
class c1,c2,c3 blue
class a1,a2,a3 purple
class o green
```

Figure 10: ResNet stacks residual blocks; each block’s skip connection
lets gradients flow directly to earlier layers.

</div>

### U-Nets

U-NET is a convolution based neural network architecture orginally
developed for biomedical image segmentation tasks. It has an
encoder-decoder architecture with skip connections in between them.

<div id="fig-u-nets">

``` mermaid
flowchart TB
    inp(("`input`")) --> e1["`Enc 1`"]
    e1 --> e2["`Enc 2`"]
    e2 --> e3["`Enc 3`"]
    e3 --> b["`bottleneck`"]
    b --> d3["`Dec 3`"]
    d3 --> d2["`Dec 2`"]
    d2 --> d1["`Dec 1`"]
    d1 --> outp(("`segmentation<br/>map`"))
    e1 -.->|"`skip`"| d1
    e2 -.->|"`skip`"| d2
    e3 -.->|"`skip`"| d3
classDef block fill:#CCCCCC02,stroke:#838383,stroke-width:1px,color:#838383
classDef red fill:#ff8181,stroke:#333,stroke-width:1px,color:#000
classDef blue fill:#7DCAFF,stroke:#333,stroke-width:1px,color:#000
classDef yellow fill:#FFFF7F,stroke:#333,stroke-width:1px,color:#000
classDef green fill:#98E6A5,stroke:#333,stroke-width:1px,color:#000
classDef purple fill:#FFCBE6,stroke:#333,stroke-width:1px,color:#000
class inp red
class e1,e2,e3 blue
class b purple
class d3,d2,d1 yellow
class outp green
```

Figure 11: U-Net’s encoder–decoder (“U”) shape with **skip connections**
copying encoder features across to the matching decoder stage.

</div>

### ViTs

You’ll learn about language models today, which use “transformer”
models. There has been some success applying transformers to images
(“vision transformers”). To make images sequential, they are split into
patches and flattened. Then apply linear embeddings and positional
embeddings and feed it to a encoder-based transformer model.

<div id="fig-vits">

![](images/ViT.gif)

Figure 12: Image credit: [Google
Blog](https://ai.googleblog.com/2020/12/transformers-for-image-recognition-at.html)

</div>
