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

[![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/saforem2/intro-hpc-bootcamp/blob/main/docs/01-neural-networks/1-mnist/index.ipynb)
[![](https://img.shields.io/badge/-View%20on%20GitHub-333333?style=flat&logo=github&labelColor=gray.png)](https://github.com/saforem2/intro-hpc-bootcamp/blob/main/docs/01-neural-networks/1-mnist/README.md)

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

    [2026-07-24 08:52:45,140872][I][ipykernel_59977/3921772995:1:<module>] MNIST data loaded: train=48000 examples validation=12000 examples test=10000 examples input shape=torch.Size([1, 28, 28])

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

``` mermaid
flowchart LR
    subgraph IN["`Input pixels **x**<br/>(28×28 = 784)`"]
        direction TB
        x1(("`x₁`"))
        x2(("`x₂`"))
        xd(("`⋮`"))
        xn(("`x₇₈₄`"))
    end
    subgraph OUT["`Output logits<br/>(10 classes)`"]
        direction TB
        y0(("`0`"))
        y1(("`⋮`"))
        y9(("`9`"))
    end
    x1 & x2 & xd & xn -->|"`W, b`"| y0 & y1 & y9
classDef block fill:#CCCCCC02,stroke:#838383,stroke-width:1px,color:#838383
classDef red fill:#ff8181,stroke:#333,stroke-width:1px,color:#000
classDef green fill:#98E6A5,stroke:#333,stroke-width:1px,color:#000
class IN,OUT block
class x1,x2,xd,xn red
class y0,y1,y9 green
```

Figure 2: A fully-connected linear layer: every input pixel connects to
every output class via $\mathbf{x}\mathbf{W} + \mathbf{b}$.

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

    [2026-07-24 08:52:45,294314][I][ipykernel_59977/2844520859:2:<module>] LinearClassifier(
      (flatten): Flatten(start_dim=1, end_dim=-1)
      (layer_1): Linear(in_features=784, out_features=10, bias=True)
    )

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

    [2026-07-24 08:52:47,028878][I][./<timed exec>:10:<module>] Epoch 0: training loss: 0.5013451065222422, accuracy: 87.66874999999999
    [2026-07-24 08:52:47,227420][I][./<timed exec>:15:<module>] Epoch 0: val. loss: 0.4935248221488709, val. accuracy: 87.6
    [2026-07-24 08:52:48,954245][I][./<timed exec>:10:<module>] Epoch 1: training loss: 0.421256408850352, accuracy: 89.04791666666667
    [2026-07-24 08:52:49,148130][I][./<timed exec>:15:<module>] Epoch 1: val. loss: 0.4117542800117046, val. accuracy: 88.95833333333333
    [2026-07-24 08:52:50,965357][I][./<timed exec>:10:<module>] Epoch 2: training loss: 0.3874150522152583, accuracy: 89.71666666666667
    [2026-07-24 08:52:51,198088][I][./<timed exec>:15:<module>] Epoch 2: val. loss: 0.3773218404422415, val. accuracy: 89.49166666666667
    [2026-07-24 08:52:53,019764][I][./<timed exec>:10:<module>] Epoch 3: training loss: 0.367519397854805, accuracy: 90.14583333333334
    [2026-07-24 08:52:53,236111][I][./<timed exec>:15:<module>] Epoch 3: val. loss: 0.35724003692256645, val. accuracy: 89.875
    [2026-07-24 08:52:55,020041][I][./<timed exec>:10:<module>] Epoch 4: training loss: 0.35398357252279916, accuracy: 90.45208333333333
    [2026-07-24 08:52:55,218616][I][./<timed exec>:15:<module>] Epoch 4: val. loss: 0.3437096652515391, val. accuracy: 90.225
    CPU times: user 9.28 s, sys: 563 ms, total: 9.84 s
    Wall time: 9.91 s

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

    [2026-07-24 08:52:55,533591][I][ipykernel_59977/372756021:2:<module>] Test loss: 0.3321311667561531, test accuracy: 90.86

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

    [2026-07-24 08:52:55,544748][I][ipykernel_59977/2368214845:8:show_failures] Showing max 10 first failures.
    [2026-07-24 08:52:55,545382][I][ipykernel_59977/2368214845:11:show_failures] The predicted class is shown first and the correct class in parentheses.

![](index_files/figure-commonmark/cell-19-output-2.png)

## Multilayer Model

Our linear model isn’t enough for high accuracy on this dataset. To
improve the model, we often need to add more layers and nonlinearities.

<div id="fig-shallow-nn">

``` mermaid
flowchart LR
    subgraph IN["`Input **x**`"]
        direction TB
        i1(("`x₁`"))
        i2(("`x₂`"))
        i3(("`x₃`"))
    end
    subgraph H["`Hidden layer<br/>σ₁(xW₁ + b₁)`"]
        direction TB
        h1(("` `"))
        h2(("` `"))
        h3(("` `"))
        h4(("` `"))
    end
    subgraph OUT["`Output<br/>σ₂(·W₂ + b₂)`"]
        direction TB
        o1(("`û`"))
    end
    i1 & i2 & i3 --> h1 & h2 & h3 & h4
    h1 & h2 & h3 & h4 --> o1
classDef block fill:#CCCCCC02,stroke:#838383,stroke-width:1px,color:#838383
classDef red fill:#ff8181,stroke:#333,stroke-width:1px,color:#000
classDef blue fill:#7DCAFF,stroke:#333,stroke-width:1px,color:#000
classDef green fill:#98E6A5,stroke:#333,stroke-width:1px,color:#000
class IN,H,OUT block
class i1,i2,i3 red
class h1,h2,h3,h4 blue
class o1 green
```

Figure 3: A shallow (single-hidden-layer) neural network.

</div>

The output of this NN can be written as

$$\begin{equation}
  \hat{u}(x) = \sigma_2(\sigma_1(\mathbf{x}\mathbf{W}_1 + \mathbf{b}_1)\mathbf{W}_2 + \mathbf{b}_2),
\end{equation}$$

where $\mathbf{x}$ is the input, $\mathbf{W}_j$ are the weights of the
neural network, $\sigma_j$ the (nonlinear) activation functions, and
$\mathbf{b}_j$ its biases.

A few of the most common activation functions, plotted over
$x \in [-5, 5]$:

``` python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 400)

def sigmoid(x): return 1 / (1 + np.exp(-x))

activations = {
    "Identity":    x,
    "Sigmoid":     sigmoid(x),
    "Tanh":        np.tanh(x),
    "ReLU":        np.maximum(0, x),
    "Leaky ReLU":  np.where(x > 0, x, 0.1 * x),
    "PReLU (a=.25)": np.where(x > 0, x, 0.25 * x),
    "ELU":         np.where(x > 0, x, np.exp(x) - 1),
    "SELU":        1.0507 * np.where(x > 0, x, 1.6733 * (np.exp(x) - 1)),
    "Softplus":    np.log1p(np.exp(x)),
    "GELU":        0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3))),
    "SiLU / Swish": x * sigmoid(x),
    "Mish":        x * np.tanh(np.log1p(np.exp(x))),
}

fig, axes = plt.subplots(3, 4, figsize=(12, 8), sharex=True, sharey=True)
for ax, (name, y) in zip(axes.flat, activations.items()):
    ax.plot(x, y, color="#7DCAFF", linewidth=2)
    ax.axhline(0, color="#838383", linewidth=0.7, zorder=0)
    ax.axvline(0, color="#838383", linewidth=0.7, zorder=0)
    ax.set_title(name, fontsize=11)
    ax.set_xlim(-5, 5)      # curves fill the full axis (no ambivalent x-margin)
    ax.set_ylim(-3, 5)      # symmetric-ish range that contains every curve
    ax.set_xticks([-5, 0, 5])
    ax.set_yticks([-2, 0, 2, 4])
    ax.grid(alpha=0.15)
fig.suptitle("Activation functions", fontsize=14)
fig.tight_layout()
plt.show()
```

<div id="fig-activation">

![](index_files/figure-commonmark/fig-activation-output-1.png)

Figure 4: Common activation functions used in neural networks.

</div>

The **activation function** introduces the nonlinearity that lets the
network learn complex tasks. Desirable properties include being
differentiable, (mostly) monotonic, and well-behaved gradients. `ReLU`
and its variants (`Leaky ReLU`, `ELU`, `GELU`, `SiLU`) dominate modern
deep networks because they avoid the vanishing-gradient problem that
plagues the saturating `Sigmoid`/`Tanh`.

Stacking several hidden layers gives a **deep** neural network — each
layer transforms the previous layer’s output, letting the model build up
increasingly abstract representations:

<div id="fig-nn-annotated">

``` mermaid
flowchart LR
    subgraph IN["`Input`"]
        direction TB
        i1(("` `"))
        i2(("` `"))
        i3(("` `"))
    end
    subgraph H1["`Hidden 1`"]
        direction TB
        a1(("` `"))
        a2(("` `"))
        a3(("` `"))
        a4(("` `"))
    end
    subgraph H2["`Hidden 2`"]
        direction TB
        b1(("` `"))
        b2(("` `"))
        b3(("` `"))
        b4(("` `"))
    end
    subgraph H3["`Hidden 3`"]
        direction TB
        c1(("` `"))
        c2(("` `"))
        c3(("` `"))
    end
    subgraph OUT["`Output`"]
        direction TB
        o1(("` `"))
        o2(("` `"))
    end
    i1 & i2 & i3 --> a1 & a2 & a3 & a4
    a1 & a2 & a3 & a4 --> b1 & b2 & b3 & b4
    b1 & b2 & b3 & b4 --> c1 & c2 & c3
    c1 & c2 & c3 --> o1 & o2
classDef block fill:#CCCCCC02,stroke:#838383,stroke-width:1px,color:#838383
classDef red fill:#ff8181,stroke:#333,stroke-width:1px,color:#000
classDef blue fill:#7DCAFF,stroke:#333,stroke-width:1px,color:#000
classDef yellow fill:#FFFF7F,stroke:#333,stroke-width:1px,color:#000
classDef purple fill:#FFCBE6,stroke:#333,stroke-width:1px,color:#000
classDef green fill:#98E6A5,stroke:#333,stroke-width:1px,color:#000
class IN,H1,H2,H3,OUT block
class i1,i2,i3 red
class a1,a2,a3,a4 blue
class b1,b2,b3,b4 yellow
class c1,c2,c3 purple
class o1,o2 green
```

Figure 5: A deep neural network with several hidden layers.

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

    [2026-07-24 08:52:57,742110][I][./<timed exec>:10:<module>] Epoch 0: training loss: 0.6950037430127461, accuracy: 80.75833333333333
    [2026-07-24 08:52:57,952408][I][./<timed exec>:15:<module>] Epoch 0: val. loss: 0.6886971491448423, val. accuracy: 80.65
    [2026-07-24 08:52:59,755951][I][./<timed exec>:10:<module>] Epoch 1: training loss: 0.3955140495697657, accuracy: 89.0
    [2026-07-24 08:52:59,988018][I][./<timed exec>:15:<module>] Epoch 1: val. loss: 0.3900272332607432, val. accuracy: 88.825
    [2026-07-24 08:53:01,816307][I][./<timed exec>:10:<module>] Epoch 2: training loss: 0.3064884059826533, accuracy: 91.19375
    [2026-07-24 08:53:02,025492][I][./<timed exec>:15:<module>] Epoch 2: val. loss: 0.30274571732003636, val. accuracy: 91.05
    [2026-07-24 08:53:03,986364][I][./<timed exec>:10:<module>] Epoch 3: training loss: 0.24886448953549067, accuracy: 92.88333333333333
    [2026-07-24 08:53:04,192434][I][./<timed exec>:15:<module>] Epoch 3: val. loss: 0.2474212978590042, val. accuracy: 92.63333333333334
    [2026-07-24 08:53:05,980319][I][./<timed exec>:10:<module>] Epoch 4: training loss: 0.2101447277466456, accuracy: 93.83749999999999
    [2026-07-24 08:53:06,182228][I][./<timed exec>:15:<module>] Epoch 4: val. loss: 0.212117029235084, val. accuracy: 93.71666666666667
    CPU times: user 9.64 s, sys: 632 ms, total: 10.3 s
    Wall time: 10.3 s

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

![](index_files/figure-commonmark/cell-24-output-1.png)

``` python
show_failures(nonlinear_model, test_dataloader)
```

    [2026-07-24 08:53:06,240455][I][ipykernel_59977/2368214845:8:show_failures] Showing max 10 first failures.
    [2026-07-24 08:53:06,241282][I][ipykernel_59977/2368214845:11:show_failures] The predicted class is shown first and the correct class in parentheses.

![](index_files/figure-commonmark/cell-25-output-2.png)

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

    [2026-07-24 08:53:07,891509][I][./<timed exec>:11:<module>] Epoch 1/6, Learning Rate: 0.1
    [2026-07-24 08:53:08,842945][I][./<timed exec>:16:<module>] Epoch 0: training loss: 0.3887119378397862, accuracy: 88.89166666666667
    [2026-07-24 08:53:09,165147][I][./<timed exec>:21:<module>] Epoch 0: val. loss: 0.3814143586556117, val. accuracy: 88.74166666666666
    [2026-07-24 08:53:12,717683][I][./<timed exec>:11:<module>] Epoch 2/6, Learning Rate: 0.010000000000000002
    [2026-07-24 08:53:13,855534][I][./<timed exec>:16:<module>] Epoch 1: training loss: 0.26931736890847485, accuracy: 91.97291666666668
    [2026-07-24 08:53:14,125009][I][./<timed exec>:21:<module>] Epoch 1: val. loss: 0.2605265693763892, val. accuracy: 92.08333333333333
    [2026-07-24 08:53:15,658634][I][./<timed exec>:11:<module>] Epoch 3/6, Learning Rate: 0.010000000000000002
    [2026-07-24 08:53:16,677409][I][./<timed exec>:16:<module>] Epoch 2: training loss: 0.23778050619487962, accuracy: 92.84791666666666
    [2026-07-24 08:53:16,921847][I][./<timed exec>:21:<module>] Epoch 2: val. loss: 0.232030248016119, val. accuracy: 93.025
    [2026-07-24 08:53:18,237662][I][./<timed exec>:11:<module>] Epoch 4/6, Learning Rate: 0.0010000000000000002
    [2026-07-24 08:53:19,418598][I][./<timed exec>:16:<module>] Epoch 3: training loss: 0.2315428444457551, accuracy: 93.05
    [2026-07-24 08:53:19,698995][I][./<timed exec>:21:<module>] Epoch 3: val. loss: 0.22659528712928295, val. accuracy: 93.11666666666667
    [2026-07-24 08:53:21,283382][I][./<timed exec>:11:<module>] Epoch 5/6, Learning Rate: 0.0010000000000000002
    [2026-07-24 08:53:22,248317][I][./<timed exec>:16:<module>] Epoch 4: training loss: 0.22928156545758246, accuracy: 93.13749999999999
    [2026-07-24 08:53:22,474206][I][./<timed exec>:21:<module>] Epoch 4: val. loss: 0.2244338550120592, val. accuracy: 93.26666666666667
    [2026-07-24 08:53:23,920003][I][./<timed exec>:11:<module>] Epoch 6/6, Learning Rate: 0.00010000000000000003
    [2026-07-24 08:53:24,900391][I][./<timed exec>:16:<module>] Epoch 5: training loss: 0.2285127173103392, accuracy: 93.14375
    [2026-07-24 08:53:25,173034][I][./<timed exec>:21:<module>] Epoch 5: val. loss: 0.22369361145297686, val. accuracy: 93.24166666666667
    CPU times: user 15.6 s, sys: 2.32 s, total: 17.9 s
    Wall time: 18.6 s
