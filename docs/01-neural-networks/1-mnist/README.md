# Intro to NNs: MNIST
Sam Foreman, Marieme Ngom, Huihuo Zheng, Bethany Lusch, Taylor Childers
2025-07-17

<link rel="preconnect" href="https://fonts.googleapis.com">

- [The MNIST dataset](#the-mnist-dataset)
- [Generalities:](#generalities)
- [Linear Model](#linear-model)
- [Learning](#learning)
- [Prediction](#prediction)
- [Multilayer Model](#multilayer-model)
- [Important things to know](#important-things-to-know)
- [Recap](#recap)
- [Homework](#homework)
- [Homework solution](#homework-solution)

[![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/saforem2/intro-hpc-bootcamp-2025/blob/main/docs/01-neural-networks/1-mnist/index.ipynb)
[![](https://img.shields.io/badge/-View%20on%20GitHub-333333?style=flat&logo=github&labelColor=gray.png)](https://github.com/saforem2/intro-hpc-bootcamp-2025/blob/main/docs/01-neural-networks/1-mnist/README.md)

> [!NOTE]
>
> Content for this tutorial has been modified from content originally
> written by:
>
> Marieme Ngom, Bethany Lusch, Asad Khan, Prasanna Balaprakash, Taylor
> Childers, Corey Adams, Kyle Felker, and Tanwi Mallick

This tutorial will serve as a gentle introduction to neural networks and
deep learning through a hands-on classification problem using the MNIST
dataset.

In particular, we will introduce neural networks and how to train and
improve their learning capabilities. We will use the PyTorch Python
library.

The [MNIST dataset](http://yann.lecun.com/exdb/mnist/) contains
thousands of examples of handwritten numbers, with each digit labeled
0-9.

<div id="fig-mnist-task">

<img src="../images/mnist_task.png" width="400" />

Figure 1: MNIST sample

</div>

``` python
import ambivalent

import matplotlib.pyplot as plt
import seaborn as sns

import ezpz
# console = ezpz.log.get_console()
logger = ezpz.get_logger('mnist')

plt.style.use(ambivalent.STYLES['ambivalent'])
sns.set_context("notebook")
plt.rcParams["figure.figsize"] = [6.4, 4.8]
```

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">[<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">2025-08-06 14:09:41,501256</span>][<span style="color: #008000; text-decoration-color: #008000">I</span>][<span style="color: #008080; text-decoration-color: #008080; font-style: italic">ezpz</span>/<span style="color: #000080; text-decoration-color: #000080">__init__</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #800080; text-decoration-color: #800080">265</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">ezpz</span>]<span style="color: #c0c0c0; text-decoration-color: #c0c0c0"> </span>Setting logging level to <span style="color: #008000; text-decoration-color: #008000">'INFO'</span> on <span style="color: #008000; text-decoration-color: #008000">'RANK == 0'</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">[<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">2025-08-06 14:09:41,503373</span>][<span style="color: #008000; text-decoration-color: #008000">I</span>][<span style="color: #008080; text-decoration-color: #008080; font-style: italic">ezpz</span>/<span style="color: #000080; text-decoration-color: #000080">__init__</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #800080; text-decoration-color: #800080">266</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">ezpz</span>]<span style="color: #c0c0c0; text-decoration-color: #c0c0c0"> </span>Setting logging level to <span style="color: #008000; text-decoration-color: #008000">'CRITICAL'</span> on all others <span style="color: #008000; text-decoration-color: #008000">'RANK != 0'</span>
</pre>

``` python
# %matplotlib inline

import torch
import torchvision
from torch import nn

import numpy 
import matplotlib.pyplot as plt
import time
```

## The MNIST dataset

We will now download the dataset that contains handwritten digits. MNIST
is a popular dataset, so we can download it via the PyTorch library.

Note:

- `x` is for the inputs (images of handwritten digits)
- `y` is for the labels or outputs (digits 0-9)
- We are given “training” and “test” datasets.
  - Training datasets are used to fit the model.
  - Test datasets are saved until the end, when we are satisfied with
    our model, to estimate how well our model generalizes to new data.

Note that downloading it the first time might take some time.

The data is split as follows:

- 60,000 training examples, 10,000 test examples
- inputs: 1 x 28 x 28 pixels
- outputs (labels): one integer per example

``` python
training_data = torchvision.datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=torchvision.transforms.ToTensor()
)

test_data = torchvision.datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=torchvision.transforms.ToTensor()
)
```

``` python
train_size = int(0.8 * len(training_data))  # 80% for training
val_size = len(training_data) - train_size  # Remaining 20% for validation
training_data, validation_data = torch.utils.data.random_split(
    training_data,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(55)
)
```

``` python
logger.info(
    " ".join([
        f"MNIST data loaded:",
        f"train={len(training_data)} examples",
        f"validation={len(validation_data)} examples",
        f"test={len(test_data)} examples",
        f"input shape={training_data[0][0].shape}" 
    ])
)
# logger.info(f'Input shape', training_data[0][0].shape)
```

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">[<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">2025-08-06 14:09:41,648835</span>][<span style="color: #008000; text-decoration-color: #008000">I</span>][<span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_93809</span>/<span style="color: #000080; text-decoration-color: #000080">3921772995</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #800080; text-decoration-color: #800080">1</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">mnist</span>]<span style="color: #c0c0c0; text-decoration-color: #c0c0c0"> </span>MNIST data loaded: <span style="color: #0000ff; text-decoration-color: #0000ff">train</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">48000</span> examples <span style="color: #0000ff; text-decoration-color: #0000ff">validation</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">12000</span> examples <span style="color: #0000ff; text-decoration-color: #0000ff">test</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">10000</span> examples input <span style="color: #0000ff; text-decoration-color: #0000ff">shape</span>=<span style="color: #800080; text-decoration-color: #800080; font-weight: bold">torch.Size</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">([</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">28</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">28</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">])</span>
</pre>

Let’s take a closer look. Here are the first 10 training digits:

``` python
pltsize=1
# plt.figure(figsize=(10*pltsize, pltsize))

for i in range(10):
    plt.subplot(1,10,i+1)
    plt.axis('off')
    # x, y = training_data[i]
    # plt.imshow(x.reshape(28, 28), cmap="gray")
    # x[0] is the image, x[1] is the label
    plt.imshow(
        numpy.reshape(
            training_data[i][0],
            (28, 28)
        ),
        cmap="gray"
    )
    plt.title(f"{training_data[i][1]}") 
```

![](index_files/figure-commonmark/cell-7-output-1.png)

## Generalities:

To train our classifier, we need (besides the data):

- A model that depend on parameters $\mathbf{\theta}$. Here we are going
  to use neural networks.
- A loss function $J(\mathbf{\theta})$ to measure the capabilities of
  the model.
- An optimization method.

## Linear Model

Let’s begin with a simple linear model: linear regression, like last
week.

We add one complication: each example is a vector (flattened image), so
the “slope” multiplication becomes a dot product. If the target output
is a vector as well, then the multiplication becomes matrix
multiplication.

Note, like before, we consider multiple examples at once, adding another
dimension to the input.

<div id="fig-linear-svg">

![](../../assets/linear-net-with-weights.svg)

Figure 2: Fully connected linear net

</div>

The linear layers in PyTorch perform a basic $xW + b$.

These “fully connected” layers connect each input to each output with
some weight parameter.

We wouldn’t expect a simple linear model $f(x) = xW+b$ directly
outputting the class label and minimizing mean squared error to work
well - the model would output labels like 3.55 and 2.11 instead of
skipping to integers.

We now need:

- A loss function $J(\theta)$ where $\theta$ is the list of parameters
  (here W and b). Last week, we used mean squared error (MSE), but this
  week let’s make two changes that make more sense for classification:
  - Change the output to be a length-10 vector of class probabilities (0
    to 1, adding to 1).
  - Cross entropy as the loss function, which is typical for
    classification. You can read more
    [here](https://gombru.github.io/2018/05/23/cross_entropy_loss/).
- An optimization method or optimizer such as the stochastic gradient
  descent (sgd) method, the Adam optimizer, RMSprop, Adagrad etc. Let’s
  start with stochastic gradient descent (sgd), like last week. For far
  more information about more advanced optimizers than basic SGD, with
  some cool animations, see
  <https://ruder.io/optimizing-gradient-descent/> or
  <https://distill.pub/2017/momentum/>.
- A learning rate. As we learned last week, the learning rate controls
  how far we move during each step.

``` python
class LinearClassifier(nn.Module):

    def __init__(self):
        super().__init__()
        # First, we need to convert the input image to a vector by using 
        # nn.Flatten(). For MNIST, it means the second dimension 28*28 becomes 784.
        self.flatten = nn.Flatten()
        # Here, we add a fully connected ("dense") layer that has 28 x 28 = 784 input nodes 
        #(one for each pixel in the input image) and 10 output nodes (for probabilities of each class).
        self.layer_1 = nn.Linear(28*28, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.layer_1(x)
        return x
```

``` python
linear_model = LinearClassifier()
logger.info(linear_model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(linear_model.parameters(), lr=0.05)
```

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">[<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">2025-08-06 14:09:41,792969</span>][<span style="color: #008000; text-decoration-color: #008000">I</span>][<span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_93809</span>/<span style="color: #000080; text-decoration-color: #000080">2844520859</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #800080; text-decoration-color: #800080">2</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">mnist</span>]<span style="color: #c0c0c0; text-decoration-color: #c0c0c0"> </span><span style="color: #800080; text-decoration-color: #800080; font-weight: bold">LinearClassifier</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">(</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">(</span>flatten<span style="color: #ff00ff; text-decoration-color: #ff00ff">)</span>: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">Flatten</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">(</span><span style="color: #0000ff; text-decoration-color: #0000ff">start_dim</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>, <span style="color: #0000ff; text-decoration-color: #0000ff">end_dim</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">-1</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">)</span>
  <span style="color: #ff00ff; text-decoration-color: #ff00ff">(</span>layer_1<span style="color: #ff00ff; text-decoration-color: #ff00ff">)</span>: <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">Linear</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">(</span><span style="color: #0000ff; text-decoration-color: #0000ff">in_features</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">784</span>, <span style="color: #0000ff; text-decoration-color: #0000ff">out_features</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">10</span>, <span style="color: #0000ff; text-decoration-color: #0000ff">bias</span>=<span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">True</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">)</span>
<span style="color: #ff00ff; text-decoration-color: #ff00ff">)</span>
</pre>

## Learning

Now we are ready to train our first model.

A training step is comprised of:

- A forward pass: the input is passed through the network
- Backpropagation: A backward pass to compute the gradient
  $\frac{\partial J}{\partial \mathbf{W}}$ of the loss function with
  respect to the parameters of the network.
- Weight updates
  $\mathbf{W} = \mathbf{W} - \alpha \frac{\partial J}{\partial \mathbf{W}}$
  where $\alpha$ is the learning rate.

How many steps do we take?

- The batch size corresponds to the number of training examples in one
  pass (forward + backward).
  - A smaller batch size allows the model to learn from individual
    examples but takes longer to train.
  - A larger batch size requires fewer steps but may result in the model
    not capturing the nuances in the data.
- The higher the batch size, the more memory you will require.
- An epoch means one pass through the whole training data (looping over
  the batches). Using few epochs can lead to underfitting and using too
  many can lead to overfitting.
- The choice of batch size and learning rate are important for
  performance, generalization and accuracy in deep learning.

``` python
batch_size = 128

# The dataloader makes our dataset iterable 
train_dataloader = torch.utils.data.DataLoader(training_data, batch_size=batch_size)
val_dataloader = torch.utils.data.DataLoader(validation_data, batch_size=batch_size)
```

``` python
def train_one_epoch(dataloader, model, loss_fn, optimizer):
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # forward pass
        pred = model(X)
        loss = loss_fn(pred, y)
        # backward pass calculates gradients
        loss.backward()
        # take one step with these gradients
        optimizer.step()
        # resets the gradients 
        optimizer.zero_grad()
```

``` python
def evaluate(dataloader, model, loss_fn):
    # Set the model to evaluation mode - some NN pieces behave differently during training
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    loss, correct = 0, 0

    # We can save computation and memory by not calculating gradients here - we aren't optimizing 
    with torch.no_grad():
        # loop over all of the batches
        for X, y in dataloader:
            pred = model(X)
            loss += loss_fn(pred, y).item()
            # how many are correct in this batch? Tracking for accuracy 
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    loss /= num_batches
    correct /= size

    accuracy = 100*correct
    return accuracy, loss
```

``` python
%%time

epochs = 5
train_acc_all = []
val_acc_all = []
for j in range(epochs):
    train_one_epoch(train_dataloader, linear_model, loss_fn, optimizer)

    # checking on the training loss and accuracy once per epoch
    acc, loss = evaluate(train_dataloader, linear_model, loss_fn)
    train_acc_all.append(acc)
    logger.info(f"Epoch {j}: training loss: {loss}, accuracy: {acc}")

    # checking on the validation loss and accuracy once per epoch
    val_acc, val_loss = evaluate(val_dataloader, linear_model, loss_fn)
    val_acc_all.append(val_acc)
    logger.info(f"Epoch {j}: val. loss: {val_loss}, val. accuracy: {val_acc}")
```

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">[<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">2025-08-06 14:09:43,891568</span>][<span style="color: #008000; text-decoration-color: #008000">I</span>][<span style="color: #008080; text-decoration-color: #008080; font-style: italic">.</span>/<span style="color: #000080; text-decoration-color: #000080">&lt;timed exec&gt;</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #800080; text-decoration-color: #800080">10</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">mnist</span>]<span style="color: #c0c0c0; text-decoration-color: #c0c0c0"> </span>Epoch <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>: training loss: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.501736267487208</span>, accuracy: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">87.56666666666668</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">[<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">2025-08-06 14:09:44,140519</span>][<span style="color: #008000; text-decoration-color: #008000">I</span>][<span style="color: #008080; text-decoration-color: #008080; font-style: italic">.</span>/<span style="color: #000080; text-decoration-color: #000080">&lt;timed exec&gt;</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #800080; text-decoration-color: #800080">15</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">mnist</span>]<span style="color: #c0c0c0; text-decoration-color: #c0c0c0"> </span>Epoch <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>: val. loss: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.49366769606762745</span>, val. accuracy: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">87.65833333333333</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">[<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">2025-08-06 14:09:46,270246</span>][<span style="color: #008000; text-decoration-color: #008000">I</span>][<span style="color: #008080; text-decoration-color: #008080; font-style: italic">.</span>/<span style="color: #000080; text-decoration-color: #000080">&lt;timed exec&gt;</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #800080; text-decoration-color: #800080">10</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">mnist</span>]<span style="color: #c0c0c0; text-decoration-color: #c0c0c0"> </span>Epoch <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>: training loss: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.4215268633762995</span>, accuracy: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">89.03750000000001</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">[<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">2025-08-06 14:09:46,590069</span>][<span style="color: #008000; text-decoration-color: #008000">I</span>][<span style="color: #008080; text-decoration-color: #008080; font-style: italic">.</span>/<span style="color: #000080; text-decoration-color: #000080">&lt;timed exec&gt;</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #800080; text-decoration-color: #800080">15</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">mnist</span>]<span style="color: #c0c0c0; text-decoration-color: #c0c0c0"> </span>Epoch <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>: val. loss: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.412016454212209</span>, val. accuracy: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">88.89166666666667</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">[<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">2025-08-06 14:09:48,687819</span>][<span style="color: #008000; text-decoration-color: #008000">I</span>][<span style="color: #008080; text-decoration-color: #008080; font-style: italic">.</span>/<span style="color: #000080; text-decoration-color: #000080">&lt;timed exec&gt;</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #800080; text-decoration-color: #800080">10</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">mnist</span>]<span style="color: #c0c0c0; text-decoration-color: #c0c0c0"> </span>Epoch <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span>: training loss: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.387637225151062</span>, accuracy: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">89.7</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">[<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">2025-08-06 14:09:48,942520</span>][<span style="color: #008000; text-decoration-color: #008000">I</span>][<span style="color: #008080; text-decoration-color: #008080; font-style: italic">.</span>/<span style="color: #000080; text-decoration-color: #000080">&lt;timed exec&gt;</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #800080; text-decoration-color: #800080">15</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">mnist</span>]<span style="color: #c0c0c0; text-decoration-color: #c0c0c0"> </span>Epoch <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span>: val. loss: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.37760588352350477</span>, val. accuracy: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">89.575</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">[<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">2025-08-06 14:09:51,053912</span>][<span style="color: #008000; text-decoration-color: #008000">I</span>][<span style="color: #008080; text-decoration-color: #008080; font-style: italic">.</span>/<span style="color: #000080; text-decoration-color: #000080">&lt;timed exec&gt;</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #800080; text-decoration-color: #800080">10</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">mnist</span>]<span style="color: #c0c0c0; text-decoration-color: #c0c0c0"> </span>Epoch <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">3</span>: training loss: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.36771197628974917</span>, accuracy: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">90.11875</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">[<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">2025-08-06 14:09:51,313019</span>][<span style="color: #008000; text-decoration-color: #008000">I</span>][<span style="color: #008080; text-decoration-color: #008080; font-style: italic">.</span>/<span style="color: #000080; text-decoration-color: #000080">&lt;timed exec&gt;</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #800080; text-decoration-color: #800080">15</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">mnist</span>]<span style="color: #c0c0c0; text-decoration-color: #c0c0c0"> </span>Epoch <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">3</span>: val. loss: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.35751901836471356</span>, val. accuracy: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">89.99166666666667</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">[<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">2025-08-06 14:09:53,420448</span>][<span style="color: #008000; text-decoration-color: #008000">I</span>][<span style="color: #008080; text-decoration-color: #008080; font-style: italic">.</span>/<span style="color: #000080; text-decoration-color: #000080">&lt;timed exec&gt;</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #800080; text-decoration-color: #800080">10</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">mnist</span>]<span style="color: #c0c0c0; text-decoration-color: #c0c0c0"> </span>Epoch <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">4</span>: training loss: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.3541540491978327</span>, accuracy: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">90.39583333333333</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">[<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">2025-08-06 14:09:53,675404</span>][<span style="color: #008000; text-decoration-color: #008000">I</span>][<span style="color: #008080; text-decoration-color: #008080; font-style: italic">.</span>/<span style="color: #000080; text-decoration-color: #000080">&lt;timed exec&gt;</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #800080; text-decoration-color: #800080">15</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">mnist</span>]<span style="color: #c0c0c0; text-decoration-color: #c0c0c0"> </span>Epoch <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">4</span>: val. loss: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.34397438469719377</span>, val. accuracy: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">90.275</span>
</pre>

    CPU times: user 11.4 s, sys: 741 ms, total: 12.2 s
    Wall time: 11.9 s

``` python
plt.figure()
plt.plot(range(epochs), train_acc_all, label='Training Acc.' )
plt.plot(range(epochs), val_acc_all, label='Validation Acc.' )
plt.xlabel('Epoch #')
plt.ylabel('Loss')
plt.legend()
```

![](index_files/figure-commonmark/cell-14-output-1.png)

``` python
# Visualize how the model is doing on the first 10 examples
pltsize=1
plt.figure(figsize=(10*pltsize, pltsize))
linear_model.eval()
batch = next(iter(train_dataloader))
predictions = linear_model(batch[0])

for i in range(10):
    plt.subplot(1,10,i+1)
    plt.axis('off')
    plt.imshow(batch[0][i,0,:,:], cmap="gray")
    plt.title('%d' % predictions[i,:].argmax())
```

![](index_files/figure-commonmark/cell-15-output-1.png)

Exercise: How can you improve the accuracy? Some things you might
consider: increasing the number of epochs, changing the learning rate,
etc.

## Prediction

Let’s see how our model generalizes to the unseen test data.

``` python
#For HW: cell to change batch size
#create dataloader for test data
# The dataloader makes our dataset iterable

batch_size_test = 256 
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size_test)
```

``` python
acc_test, loss_test = evaluate(test_dataloader, linear_model, loss_fn)
logger.info(f"Test loss: {loss_test}, test accuracy: {acc_test}")
# logger.info("Test loss: %.4f, test accuracy: %.2f%%" % (loss_test, acc_test))
```

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">[<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">2025-08-06 14:09:54,063938</span>][<span style="color: #008000; text-decoration-color: #008000">I</span>][<span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_93809</span>/<span style="color: #000080; text-decoration-color: #000080">372756021</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #800080; text-decoration-color: #800080">2</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">mnist</span>]<span style="color: #c0c0c0; text-decoration-color: #c0c0c0"> </span>Test loss: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.3321808374486864</span>, test accuracy: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">90.84</span>
</pre>

We can now take a closer look at the results.

Let’s define a helper function to show the failure cases of our
classifier.

``` python
def show_failures(model, dataloader, maxtoshow=10):
    model.eval()
    batch = next(iter(dataloader))
    predictions = model(batch[0])

    rounded = predictions.argmax(1)
    errors = rounded!=batch[1]
    logger.info(
        f"Showing max {maxtoshow} first failures."
    )
    logger.info("The predicted class is shown first and the correct class in parentheses.")
    ii = 0
    plt.figure(figsize=(maxtoshow, 1))
    for i in range(batch[0].shape[0]):
        if ii>=maxtoshow:
            break
        if errors[i]:
            plt.subplot(1, maxtoshow, ii+1)
            plt.axis('off')
            plt.imshow(batch[0][i,0,:,:], cmap="gray")
            plt.title("%d (%d)" % (rounded[i], batch[1][i]))
            ii = ii + 1
```

Here are the first 10 images from the test data that this small model
classified to a wrong class:

``` python
show_failures(linear_model, test_dataloader)
```

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">[<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">2025-08-06 14:09:54,078228</span>][<span style="color: #008000; text-decoration-color: #008000">I</span>][<span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_93809</span>/<span style="color: #000080; text-decoration-color: #000080">2368214845</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #800080; text-decoration-color: #800080">8</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">mnist</span>]<span style="color: #c0c0c0; text-decoration-color: #c0c0c0"> </span>Showing max <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">10</span> first failures.
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">[<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">2025-08-06 14:09:54,078872</span>][<span style="color: #008000; text-decoration-color: #008000">I</span>][<span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_93809</span>/<span style="color: #000080; text-decoration-color: #000080">2368214845</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #800080; text-decoration-color: #800080">11</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">mnist</span>]<span style="color: #c0c0c0; text-decoration-color: #c0c0c0"> </span>The predicted class is shown first and the correct class in parentheses.
</pre>

![](index_files/figure-commonmark/cell-19-output-3.png)

## Multilayer Model

Our linear model isn’t enough for high accuracy on this dataset. To
improve the model, we often need to add more layers and nonlinearities.

<div id="fig-shallow-nn">

![](../images/shallow_nn.png)

Figure 3: Shallow neural network

</div>

The output of this NN can be written as

$$\begin{equation}
  \hat{u}(x) = \sigma_2(\sigma_1(\mathbf{x}\mathbf{W}_1 + \mathbf{b}_1)\mathbf{W}_2 + \mathbf{b}_2),
\end{equation}$$

where $\mathbf{x}$ is the input, $\mathbf{W}_j$ are the weights of the
neural network, $\sigma_j$ the (nonlinear) activation functions, and
$\mathbf{b}_j$ its biases. The activation function introduces the
nonlinearity and makes it possible to learn more complex tasks.
Desirable properties in an activation function include being
differentiable, bounded, and monotonic.

Image source:
[PragatiBaheti](https://www.v7labs.com/blog/neural-networks-activation-functions)

<div id="fig-activation">

![](../images/activation.jpeg)

Figure 4: Activation functions

</div>

Adding more layers to obtain a deep neural network:

<div id="fig-nn-annotated">

![](../images/deep_nn_annotated.jpg)

Figure 5

</div>

## Important things to know

Deep Neural networks can be overly flexible/complicated and “overfit”
your data, just like fitting overly complicated polynomials:

<div id="fig-bias-variance">

![](../images/bias_vs_variance.png)

Figure 6: Bias-variance tradeoff

</div>

Vizualization wrt to the accuracy and loss (Image source:
[Baeldung](https://www.baeldung.com/cs/ml-underfitting-overfitting)):

<div id="fig-acc-under-over">

![](./images/acc_under_over.webp)

Figure 7: Visualization of accuracy and loss

</div>

To improve the generalization of our model on previously unseen data, we
employ a technique known as regularization, which constrains our
optimization problem in order to discourage complex models.

- Dropout is the commonly used regularization technique. The Dropout
  layer randomly sets input units to 0 with a frequency of rate at each
  step during training time, which helps prevent overfitting.
- Penalizing the loss function by adding a term such as
  $\lambda ||\mathbf{W}||^2$ is alsp a commonly used regularization
  technique. This helps “control” the magnitude of the weights of the
  network.

Vanishing gradients  
Gradients become small as they propagate backward through the layers.

Squashing activation functions like sigmoid or tanh could cause this.

Exploding gradients  
Gradients grow exponentially usually due to “poor” weight
initialization.

We can now implement a deep network in PyTorch.

`nn.Dropout()` performs the Dropout operation mentioned earlier:

``` python
#For HW: cell to change activation
class NonlinearClassifier(nn.Module):

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layers_stack = nn.Sequential(
            nn.Linear(28*28, 50),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(50, 50),
            nn.ReLU(),
           # nn.Dropout(0.2),
            nn.Linear(50, 50),
            nn.ReLU(),
           # nn.Dropout(0.2),
            nn.Linear(50, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.layers_stack(x)

        return x
```

``` python
#### For HW: cell to change learning rate
nonlinear_model = NonlinearClassifier()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(nonlinear_model.parameters(), lr=0.05)
```

``` python
%%time

epochs = 5
train_acc_all = []
val_acc_all = []
for j in range(epochs):
    train_one_epoch(train_dataloader, nonlinear_model, loss_fn, optimizer)

    # checking on the training loss and accuracy once per epoch
    acc, loss = evaluate(train_dataloader, nonlinear_model, loss_fn)
    train_acc_all.append(acc)
    logger.info(f"Epoch {j}: training loss: {loss}, accuracy: {acc}")

    # checking on the validation loss and accuracy once per epoch
    val_acc, val_loss = evaluate(val_dataloader, nonlinear_model, loss_fn)
    val_acc_all.append(val_acc)
    logger.info(f"Epoch {j}: val. loss: {val_loss}, val. accuracy: {val_acc}")
```

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">[<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">2025-08-06 14:09:56,471780</span>][<span style="color: #008000; text-decoration-color: #008000">I</span>][<span style="color: #008080; text-decoration-color: #008080; font-style: italic">.</span>/<span style="color: #000080; text-decoration-color: #000080">&lt;timed exec&gt;</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #800080; text-decoration-color: #800080">10</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">mnist</span>]<span style="color: #c0c0c0; text-decoration-color: #c0c0c0"> </span>Epoch <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>: training loss: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.8599104603131612</span>, accuracy: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">75.42708333333333</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">[<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">2025-08-06 14:09:56,743182</span>][<span style="color: #008000; text-decoration-color: #008000">I</span>][<span style="color: #008080; text-decoration-color: #008080; font-style: italic">.</span>/<span style="color: #000080; text-decoration-color: #000080">&lt;timed exec&gt;</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #800080; text-decoration-color: #800080">15</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">mnist</span>]<span style="color: #c0c0c0; text-decoration-color: #c0c0c0"> </span>Epoch <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>: val. loss: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.8549650825084524</span>, val. accuracy: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">75.20833333333333</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">[<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">2025-08-06 14:09:59,048450</span>][<span style="color: #008000; text-decoration-color: #008000">I</span>][<span style="color: #008080; text-decoration-color: #008080; font-style: italic">.</span>/<span style="color: #000080; text-decoration-color: #000080">&lt;timed exec&gt;</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #800080; text-decoration-color: #800080">10</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">mnist</span>]<span style="color: #c0c0c0; text-decoration-color: #c0c0c0"> </span>Epoch <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>: training loss: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.38281604993343354</span>, accuracy: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">89.03541666666666</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">[<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">2025-08-06 14:09:59,309815</span>][<span style="color: #008000; text-decoration-color: #008000">I</span>][<span style="color: #008080; text-decoration-color: #008080; font-style: italic">.</span>/<span style="color: #000080; text-decoration-color: #000080">&lt;timed exec&gt;</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #800080; text-decoration-color: #800080">15</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">mnist</span>]<span style="color: #c0c0c0; text-decoration-color: #c0c0c0"> </span>Epoch <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>: val. loss: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.3738475491074806</span>, val. accuracy: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">89.06666666666668</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">[<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">2025-08-06 14:10:01,550578</span>][<span style="color: #008000; text-decoration-color: #008000">I</span>][<span style="color: #008080; text-decoration-color: #008080; font-style: italic">.</span>/<span style="color: #000080; text-decoration-color: #000080">&lt;timed exec&gt;</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #800080; text-decoration-color: #800080">10</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">mnist</span>]<span style="color: #c0c0c0; text-decoration-color: #c0c0c0"> </span>Epoch <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span>: training loss: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.2936612309217453</span>, accuracy: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">91.51458333333333</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">[<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">2025-08-06 14:10:01,816712</span>][<span style="color: #008000; text-decoration-color: #008000">I</span>][<span style="color: #008080; text-decoration-color: #008080; font-style: italic">.</span>/<span style="color: #000080; text-decoration-color: #000080">&lt;timed exec&gt;</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #800080; text-decoration-color: #800080">15</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">mnist</span>]<span style="color: #c0c0c0; text-decoration-color: #c0c0c0"> </span>Epoch <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span>: val. loss: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.28486612344041784</span>, val. accuracy: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">91.425</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">[<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">2025-08-06 14:10:04,120426</span>][<span style="color: #008000; text-decoration-color: #008000">I</span>][<span style="color: #008080; text-decoration-color: #008080; font-style: italic">.</span>/<span style="color: #000080; text-decoration-color: #000080">&lt;timed exec&gt;</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #800080; text-decoration-color: #800080">10</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">mnist</span>]<span style="color: #c0c0c0; text-decoration-color: #c0c0c0"> </span>Epoch <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">3</span>: training loss: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.24769398874044418</span>, accuracy: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">92.71875</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">[<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">2025-08-06 14:10:04,382201</span>][<span style="color: #008000; text-decoration-color: #008000">I</span>][<span style="color: #008080; text-decoration-color: #008080; font-style: italic">.</span>/<span style="color: #000080; text-decoration-color: #000080">&lt;timed exec&gt;</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #800080; text-decoration-color: #800080">15</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">mnist</span>]<span style="color: #c0c0c0; text-decoration-color: #c0c0c0"> </span>Epoch <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">3</span>: val. loss: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.24479275601024322</span>, val. accuracy: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">92.64166666666667</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">[<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">2025-08-06 14:10:06,617462</span>][<span style="color: #008000; text-decoration-color: #008000">I</span>][<span style="color: #008080; text-decoration-color: #008080; font-style: italic">.</span>/<span style="color: #000080; text-decoration-color: #000080">&lt;timed exec&gt;</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #800080; text-decoration-color: #800080">10</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">mnist</span>]<span style="color: #c0c0c0; text-decoration-color: #c0c0c0"> </span>Epoch <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">4</span>: training loss: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.21167574165264766</span>, accuracy: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">93.85</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">[<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">2025-08-06 14:10:06,878814</span>][<span style="color: #008000; text-decoration-color: #008000">I</span>][<span style="color: #008080; text-decoration-color: #008080; font-style: italic">.</span>/<span style="color: #000080; text-decoration-color: #000080">&lt;timed exec&gt;</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #800080; text-decoration-color: #800080">15</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">mnist</span>]<span style="color: #c0c0c0; text-decoration-color: #c0c0c0"> </span>Epoch <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">4</span>: val. loss: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.21088066879422107</span>, val. accuracy: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">93.60000000000001</span>
</pre>

    CPU times: user 12.2 s, sys: 913 ms, total: 13.1 s
    Wall time: 12.7 s

``` python
# pltsize=1
# plt.figure(figsize=(10*pltsize, 10 * pltsize))
plt.figure()
plt.plot(range(epochs), train_acc_all,label = 'Training Acc.' )
plt.plot(range(epochs), val_acc_all, label = 'Validation Acc.' )
plt.xlabel('Epoch #')
plt.ylabel('Loss')
plt.legend()
```

![](index_files/figure-commonmark/cell-23-output-1.png)

``` python
show_failures(nonlinear_model, test_dataloader)
```

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">[<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">2025-08-06 14:10:06,943912</span>][<span style="color: #008000; text-decoration-color: #008000">I</span>][<span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_93809</span>/<span style="color: #000080; text-decoration-color: #000080">2368214845</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #800080; text-decoration-color: #800080">8</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">mnist</span>]<span style="color: #c0c0c0; text-decoration-color: #c0c0c0"> </span>Showing max <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">10</span> first failures.
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">[<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">2025-08-06 14:10:06,944644</span>][<span style="color: #008000; text-decoration-color: #008000">I</span>][<span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_93809</span>/<span style="color: #000080; text-decoration-color: #000080">2368214845</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #800080; text-decoration-color: #800080">11</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">mnist</span>]<span style="color: #c0c0c0; text-decoration-color: #c0c0c0"> </span>The predicted class is shown first and the correct class in parentheses.
</pre>

![](index_files/figure-commonmark/cell-24-output-3.png)

## Recap

To train and validate a neural network model, you need:

- Data split into training/validation/test sets,
- A model with parameters to learn
- An appropriate loss function
- An optimizer (with tunable parameters such as learning rate, weight
  decay etc.) used to learn the parameters of the model.

## Homework

1.  Compare the quality of your model when using different:

- batch sizes
- learning rates
- activation functions

3.  Bonus: What is a learning rate scheduler?

If you have time, experiment with how to improve the model.

Note: training and validation data can be used to compare models, but
test data should be saved until the end as a final check of
generalization.

## Homework solution

Make the following changes to the cells with the comment “\#For HW”

``` python
#####################To modify the batch size##########################
batch_size = 32 # 64, 128, 256, 512

# The dataloader makes our dataset iterable 
train_dataloader = torch.utils.data.DataLoader(training_data, batch_size=batch_size)
val_dataloader = torch.utils.data.DataLoader(validation_data, batch_size=batch_size)
##############################################################################


##########################To change the learning rate##########################
optimizer = torch.optim.SGD(nonlinear_model.parameters(), lr=0.01) #modify the value of lr
##############################################################################


##########################To change activation##########################
###### Go to https://pytorch.org/docs/main/nn.html#non-linear-activations-weighted-sum-nonlinearity for more activations ######
class NonlinearClassifier(nn.Module):

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layers_stack = nn.Sequential(
            nn.Linear(28*28, 50),
            nn.Sigmoid(), #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(50, 50),
            nn.Tanh(), #nn.ReLU(),
           # nn.Dropout(0.2),
            nn.Linear(50, 50),
            nn.ReLU(),
           # nn.Dropout(0.2),
            nn.Linear(50, 10)
        )
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.layers_stack(x)

        return x
##############################################################################
```

Bonus question: A learning rate scheduler is an essential deep learning
technique used to dynamically adjust the learning rate during training.
This strategic can significantly impact the convergence speed and
overall performance of a neural network. See below on how to incorporate
it to your training.

``` python
nonlinear_model = NonlinearClassifier()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(nonlinear_model.parameters(), lr=0.1)

# Step learning rate scheduler: reduce by a factor of 0.1 every 2 epochs (only for illustrative purposes)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
```

``` python
%%time

epochs = 6
train_acc_all = []
val_acc_all = []
for j in range(epochs):
    train_one_epoch(train_dataloader, nonlinear_model, loss_fn, optimizer)
    #step the scheduler
    scheduler.step()

    # logger.info the current learning rate
    current_lr = optimizer.param_groups[0]['lr']
    logger.info(f"Epoch {j+1}/{epochs}, Learning Rate: {current_lr}")

    # checking on the training loss and accuracy once per epoch
    acc, loss = evaluate(train_dataloader, nonlinear_model, loss_fn)
    train_acc_all.append(acc)
    logger.info(f"Epoch {j}: training loss: {loss}, accuracy: {acc}")

    # checking on the validation loss and accuracy once per epoch
    val_acc, val_loss = evaluate(val_dataloader, nonlinear_model, loss_fn)
    val_acc_all.append(val_acc)
    logger.info(f"Epoch {j}: val. loss: {val_loss}, val. accuracy: {val_acc}")
```

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">[<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">2025-08-06 14:10:09,196544</span>][<span style="color: #008000; text-decoration-color: #008000">I</span>][<span style="color: #008080; text-decoration-color: #008080; font-style: italic">.</span>/<span style="color: #000080; text-decoration-color: #000080">&lt;timed exec&gt;</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #800080; text-decoration-color: #800080">11</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">mnist</span>]<span style="color: #c0c0c0; text-decoration-color: #c0c0c0"> </span>Epoch <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>/<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">6</span>, Learning Rate: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.1</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">[<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">2025-08-06 14:10:10,431889</span>][<span style="color: #008000; text-decoration-color: #008000">I</span>][<span style="color: #008080; text-decoration-color: #008080; font-style: italic">.</span>/<span style="color: #000080; text-decoration-color: #000080">&lt;timed exec&gt;</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #800080; text-decoration-color: #800080">16</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">mnist</span>]<span style="color: #c0c0c0; text-decoration-color: #c0c0c0"> </span>Epoch <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>: training loss: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.36225916573901973</span>, accuracy: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">89.51041666666667</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">[<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">2025-08-06 14:10:10,726160</span>][<span style="color: #008000; text-decoration-color: #008000">I</span>][<span style="color: #008080; text-decoration-color: #008080; font-style: italic">.</span>/<span style="color: #000080; text-decoration-color: #000080">&lt;timed exec&gt;</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #800080; text-decoration-color: #800080">21</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">mnist</span>]<span style="color: #c0c0c0; text-decoration-color: #c0c0c0"> </span>Epoch <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>: val. loss: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.3499938757220904</span>, val. accuracy: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">89.64166666666667</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">[<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">2025-08-06 14:10:12,357787</span>][<span style="color: #008000; text-decoration-color: #008000">I</span>][<span style="color: #008080; text-decoration-color: #008080; font-style: italic">.</span>/<span style="color: #000080; text-decoration-color: #000080">&lt;timed exec&gt;</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #800080; text-decoration-color: #800080">11</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">mnist</span>]<span style="color: #c0c0c0; text-decoration-color: #c0c0c0"> </span>Epoch <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span>/<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">6</span>, Learning Rate: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.010000000000000002</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">[<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">2025-08-06 14:10:13,514800</span>][<span style="color: #008000; text-decoration-color: #008000">I</span>][<span style="color: #008080; text-decoration-color: #008080; font-style: italic">.</span>/<span style="color: #000080; text-decoration-color: #000080">&lt;timed exec&gt;</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #800080; text-decoration-color: #800080">16</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">mnist</span>]<span style="color: #c0c0c0; text-decoration-color: #c0c0c0"> </span>Epoch <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>: training loss: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.25578233787293236</span>, accuracy: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">92.43333333333334</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">[<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">2025-08-06 14:10:13,804900</span>][<span style="color: #008000; text-decoration-color: #008000">I</span>][<span style="color: #008080; text-decoration-color: #008080; font-style: italic">.</span>/<span style="color: #000080; text-decoration-color: #000080">&lt;timed exec&gt;</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #800080; text-decoration-color: #800080">21</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">mnist</span>]<span style="color: #c0c0c0; text-decoration-color: #c0c0c0"> </span>Epoch <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>: val. loss: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.247510635972023</span>, val. accuracy: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">92.45833333333333</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">[<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">2025-08-06 14:10:15,442200</span>][<span style="color: #008000; text-decoration-color: #008000">I</span>][<span style="color: #008080; text-decoration-color: #008080; font-style: italic">.</span>/<span style="color: #000080; text-decoration-color: #000080">&lt;timed exec&gt;</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #800080; text-decoration-color: #800080">11</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">mnist</span>]<span style="color: #c0c0c0; text-decoration-color: #c0c0c0"> </span>Epoch <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">3</span>/<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">6</span>, Learning Rate: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.010000000000000002</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">[<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">2025-08-06 14:10:16,704868</span>][<span style="color: #008000; text-decoration-color: #008000">I</span>][<span style="color: #008080; text-decoration-color: #008080; font-style: italic">.</span>/<span style="color: #000080; text-decoration-color: #000080">&lt;timed exec&gt;</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #800080; text-decoration-color: #800080">16</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">mnist</span>]<span style="color: #c0c0c0; text-decoration-color: #c0c0c0"> </span>Epoch <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span>: training loss: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.23375169723356765</span>, accuracy: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">93.0875</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">[<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">2025-08-06 14:10:17,106111</span>][<span style="color: #008000; text-decoration-color: #008000">I</span>][<span style="color: #008080; text-decoration-color: #008080; font-style: italic">.</span>/<span style="color: #000080; text-decoration-color: #000080">&lt;timed exec&gt;</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #800080; text-decoration-color: #800080">21</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">mnist</span>]<span style="color: #c0c0c0; text-decoration-color: #c0c0c0"> </span>Epoch <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span>: val. loss: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.22843868960440158</span>, val. accuracy: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">93.01666666666667</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">[<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">2025-08-06 14:10:18,787170</span>][<span style="color: #008000; text-decoration-color: #008000">I</span>][<span style="color: #008080; text-decoration-color: #008080; font-style: italic">.</span>/<span style="color: #000080; text-decoration-color: #000080">&lt;timed exec&gt;</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #800080; text-decoration-color: #800080">11</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">mnist</span>]<span style="color: #c0c0c0; text-decoration-color: #c0c0c0"> </span>Epoch <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">4</span>/<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">6</span>, Learning Rate: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.0010000000000000002</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">[<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">2025-08-06 14:10:19,956489</span>][<span style="color: #008000; text-decoration-color: #008000">I</span>][<span style="color: #008080; text-decoration-color: #008080; font-style: italic">.</span>/<span style="color: #000080; text-decoration-color: #000080">&lt;timed exec&gt;</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #800080; text-decoration-color: #800080">16</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">mnist</span>]<span style="color: #c0c0c0; text-decoration-color: #c0c0c0"> </span>Epoch <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">3</span>: training loss: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.22560566428179543</span>, accuracy: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">93.28125</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">[<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">2025-08-06 14:10:20,248482</span>][<span style="color: #008000; text-decoration-color: #008000">I</span>][<span style="color: #008080; text-decoration-color: #008080; font-style: italic">.</span>/<span style="color: #000080; text-decoration-color: #000080">&lt;timed exec&gt;</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #800080; text-decoration-color: #800080">21</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">mnist</span>]<span style="color: #c0c0c0; text-decoration-color: #c0c0c0"> </span>Epoch <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">3</span>: val. loss: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.22097941367824872</span>, val. accuracy: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">93.24166666666667</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">[<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">2025-08-06 14:10:21,892911</span>][<span style="color: #008000; text-decoration-color: #008000">I</span>][<span style="color: #008080; text-decoration-color: #008080; font-style: italic">.</span>/<span style="color: #000080; text-decoration-color: #000080">&lt;timed exec&gt;</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #800080; text-decoration-color: #800080">11</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">mnist</span>]<span style="color: #c0c0c0; text-decoration-color: #c0c0c0"> </span>Epoch <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">5</span>/<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">6</span>, Learning Rate: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.0010000000000000002</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">[<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">2025-08-06 14:10:23,058346</span>][<span style="color: #008000; text-decoration-color: #008000">I</span>][<span style="color: #008080; text-decoration-color: #008080; font-style: italic">.</span>/<span style="color: #000080; text-decoration-color: #000080">&lt;timed exec&gt;</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #800080; text-decoration-color: #800080">16</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">mnist</span>]<span style="color: #c0c0c0; text-decoration-color: #c0c0c0"> </span>Epoch <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">4</span>: training loss: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.2246789121987919</span>, accuracy: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">93.3125</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">[<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">2025-08-06 14:10:23,349911</span>][<span style="color: #008000; text-decoration-color: #008000">I</span>][<span style="color: #008080; text-decoration-color: #008080; font-style: italic">.</span>/<span style="color: #000080; text-decoration-color: #000080">&lt;timed exec&gt;</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #800080; text-decoration-color: #800080">21</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">mnist</span>]<span style="color: #c0c0c0; text-decoration-color: #c0c0c0"> </span>Epoch <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">4</span>: val. loss: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.22016419530908266</span>, val. accuracy: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">93.2</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">[<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">2025-08-06 14:10:24,988052</span>][<span style="color: #008000; text-decoration-color: #008000">I</span>][<span style="color: #008080; text-decoration-color: #008080; font-style: italic">.</span>/<span style="color: #000080; text-decoration-color: #000080">&lt;timed exec&gt;</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #800080; text-decoration-color: #800080">11</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">mnist</span>]<span style="color: #c0c0c0; text-decoration-color: #c0c0c0"> </span>Epoch <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">6</span>/<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">6</span>, Learning Rate: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.00010000000000000003</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">[<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">2025-08-06 14:10:26,167791</span>][<span style="color: #008000; text-decoration-color: #008000">I</span>][<span style="color: #008080; text-decoration-color: #008080; font-style: italic">.</span>/<span style="color: #000080; text-decoration-color: #000080">&lt;timed exec&gt;</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #800080; text-decoration-color: #800080">16</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">mnist</span>]<span style="color: #c0c0c0; text-decoration-color: #c0c0c0"> </span>Epoch <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">5</span>: training loss: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.22391905495276054</span>, accuracy: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">93.32708333333333</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">[<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">2025-08-06 14:10:26,465866</span>][<span style="color: #008000; text-decoration-color: #008000">I</span>][<span style="color: #008080; text-decoration-color: #008080; font-style: italic">.</span>/<span style="color: #000080; text-decoration-color: #000080">&lt;timed exec&gt;</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #800080; text-decoration-color: #800080">21</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">mnist</span>]<span style="color: #c0c0c0; text-decoration-color: #c0c0c0"> </span>Epoch <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">5</span>: val. loss: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.21933106701572735</span>, val. accuracy: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">93.24166666666667</span>
</pre>

    CPU times: user 17.8 s, sys: 3.11 s, total: 20.9 s
    Wall time: 19.2 s
