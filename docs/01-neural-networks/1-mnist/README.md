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

    [2026-07-23 09:11:46,436496][I][ipykernel_8544/3921772995:1:<module>] MNIST data loaded: train=48000 examples validation=12000 examples test=10000 examples input shape=torch.Size([1, 28, 28])

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

    [2026-07-23 09:11:46,599669][I][ipykernel_8544/2844520859:2:<module>] LinearClassifier(
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

    [2026-07-23 09:11:48,444105][I][./<timed exec>:10:<module>] Epoch 0: training loss: 0.5021753667195638, accuracy: 87.54166666666666
    [2026-07-23 09:11:48,622166][I][./<timed exec>:15:<module>] Epoch 0: val. loss: 0.49439280178952727, val. accuracy: 87.575
    [2026-07-23 09:11:50,163684][I][./<timed exec>:10:<module>] Epoch 1: training loss: 0.42166710595289864, accuracy: 88.99791666666667
    [2026-07-23 09:11:50,356822][I][./<timed exec>:15:<module>] Epoch 1: val. loss: 0.4123108922801119, val. accuracy: 88.88333333333334
    [2026-07-23 09:11:52,061236][I][./<timed exec>:10:<module>] Epoch 2: training loss: 0.3876567794481913, accuracy: 89.65625
    [2026-07-23 09:11:52,291515][I][./<timed exec>:15:<module>] Epoch 2: val. loss: 0.37775340612898484, val. accuracy: 89.5
    [2026-07-23 09:11:54,489269][I][./<timed exec>:10:<module>] Epoch 3: training loss: 0.36766527581214903, accuracy: 90.09375
    [2026-07-23 09:11:54,699309][I][./<timed exec>:15:<module>] Epoch 3: val. loss: 0.3576002130483059, val. accuracy: 89.96666666666667
    [2026-07-23 09:11:56,934688][I][./<timed exec>:10:<module>] Epoch 4: training loss: 0.3540666294892629, accuracy: 90.44375
    [2026-07-23 09:11:57,152202][I][./<timed exec>:15:<module>] Epoch 4: val. loss: 0.34402238910502575, val. accuracy: 90.24166666666666
    CPU times: user 9.62 s, sys: 587 ms, total: 10.2 s
    Wall time: 10.5 s

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

    [2026-07-23 09:11:57,544530][I][ipykernel_8544/372756021:2:<module>] Test loss: 0.3318280646577477, test accuracy: 90.88000000000001

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

    [2026-07-23 09:11:57,557118][I][ipykernel_8544/2368214845:8:show_failures] Showing max 10 first failures.
    [2026-07-23 09:11:57,557903][I][ipykernel_8544/2368214845:11:show_failures] The predicted class is shown first and the correct class in parentheses.

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

fig, axes = plt.subplots(3, 4, figsize=(11, 7), sharex=True)
for ax, (name, y) in zip(axes.flat, activations.items()):
    ax.plot(x, y, color="#7DCAFF", linewidth=2)
    ax.axhline(0, color="#838383", linewidth=0.6)
    ax.axvline(0, color="#838383", linewidth=0.6)
    ax.set_title(name, fontsize=10)
    ax.set_ylim(-2, 3)
    ax.grid(alpha=0.15)
fig.suptitle("Activation functions", fontsize=13)
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

    [2026-07-23 09:12:00,695486][I][./<timed exec>:10:<module>] Epoch 0: training loss: 0.7232766718864441, accuracy: 78.77708333333334
    [2026-07-23 09:12:00,887180][I][./<timed exec>:15:<module>] Epoch 0: val. loss: 0.7121975472632874, val. accuracy: 78.525
    [2026-07-23 09:12:03,045649][I][./<timed exec>:10:<module>] Epoch 1: training loss: 0.38257885638872785, accuracy: 89.05416666666667
    [2026-07-23 09:12:03,328278][I][./<timed exec>:15:<module>] Epoch 1: val. loss: 0.3743061526341641, val. accuracy: 89.08333333333334
    [2026-07-23 09:12:05,163117][I][./<timed exec>:10:<module>] Epoch 2: training loss: 0.28928814506530764, accuracy: 91.675
    [2026-07-23 09:12:05,375846][I][./<timed exec>:15:<module>] Epoch 2: val. loss: 0.28289543060546224, val. accuracy: 91.34166666666667
    [2026-07-23 09:12:07,673633][I][./<timed exec>:10:<module>] Epoch 3: training loss: 0.23921235303084057, accuracy: 93.03333333333333
    [2026-07-23 09:12:07,905358][I][./<timed exec>:15:<module>] Epoch 3: val. loss: 0.2356560970557497, val. accuracy: 92.88333333333333
    [2026-07-23 09:12:09,613374][I][./<timed exec>:10:<module>] Epoch 4: training loss: 0.20260275808970132, accuracy: 94.1
    [2026-07-23 09:12:09,807768][I][./<timed exec>:15:<module>] Epoch 4: val. loss: 0.20290001362879226, val. accuracy: 93.875
    CPU times: user 10.6 s, sys: 657 ms, total: 11.3 s
    Wall time: 11.6 s

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

    [2026-07-23 09:12:10,038198][I][ipykernel_8544/2368214845:8:show_failures] Showing max 10 first failures.
    [2026-07-23 09:12:10,038781][I][ipykernel_8544/2368214845:11:show_failures] The predicted class is shown first and the correct class in parentheses.

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

    [2026-07-23 09:12:11,503100][I][./<timed exec>:11:<module>] Epoch 1/6, Learning Rate: 0.1
    [2026-07-23 09:12:12,426913][I][./<timed exec>:16:<module>] Epoch 0: training loss: 0.3537772993569573, accuracy: 89.78125
    [2026-07-23 09:12:12,735186][I][./<timed exec>:21:<module>] Epoch 0: val. loss: 0.3430143823822339, val. accuracy: 89.7
    [2026-07-23 09:12:14,628900][I][./<timed exec>:11:<module>] Epoch 2/6, Learning Rate: 0.010000000000000002
    [2026-07-23 09:12:15,626983][I][./<timed exec>:16:<module>] Epoch 1: training loss: 0.24273540047431985, accuracy: 92.70625
    [2026-07-23 09:12:15,842780][I][./<timed exec>:21:<module>] Epoch 1: val. loss: 0.2363814845184485, val. accuracy: 92.73333333333333
    [2026-07-23 09:12:17,302440][I][./<timed exec>:11:<module>] Epoch 3/6, Learning Rate: 0.010000000000000002
    [2026-07-23 09:12:18,312618][I][./<timed exec>:16:<module>] Epoch 2: training loss: 0.2182977637052536, accuracy: 93.51041666666666
    [2026-07-23 09:12:18,543495][I][./<timed exec>:21:<module>] Epoch 2: val. loss: 0.21473074538012346, val. accuracy: 93.53333333333333
    [2026-07-23 09:12:19,758846][I][./<timed exec>:11:<module>] Epoch 4/6, Learning Rate: 0.0010000000000000002
    [2026-07-23 09:12:20,586184][I][./<timed exec>:16:<module>] Epoch 3: training loss: 0.21202309411391615, accuracy: 93.63125
    [2026-07-23 09:12:20,794456][I][./<timed exec>:21:<module>] Epoch 3: val. loss: 0.2084656907469034, val. accuracy: 93.60833333333333
    [2026-07-23 09:12:22,042008][I][./<timed exec>:11:<module>] Epoch 5/6, Learning Rate: 0.0010000000000000002
    [2026-07-23 09:12:23,061759][I][./<timed exec>:16:<module>] Epoch 4: training loss: 0.21095506247133017, accuracy: 93.68125
    [2026-07-23 09:12:23,324557][I][./<timed exec>:21:<module>] Epoch 4: val. loss: 0.2076889000783364, val. accuracy: 93.61666666666667
    [2026-07-23 09:12:24,718408][I][./<timed exec>:11:<module>] Epoch 6/6, Learning Rate: 0.00010000000000000003
    [2026-07-23 09:12:25,720419][I][./<timed exec>:16:<module>] Epoch 5: training loss: 0.21012212329357863, accuracy: 93.70625
    [2026-07-23 09:12:25,991981][I][./<timed exec>:21:<module>] Epoch 5: val. loss: 0.20699219423532486, val. accuracy: 93.65
    CPU times: user 14.2 s, sys: 2.03 s, total: 16.2 s
    Wall time: 15.9 s
