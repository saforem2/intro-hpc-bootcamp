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

    [2026-07-22 15:42:29,147401][I][ipykernel_6525/3921772995:1:<module>] MNIST data loaded: train=48000 examples validation=12000 examples test=10000 examples input shape=torch.Size([1, 28, 28])

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

    [2026-07-22 15:42:29,289073][I][ipykernel_6525/2844520859:2:<module>] LinearClassifier(
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

    [2026-07-22 15:42:30,919100][I][./<timed exec>:10:<module>] Epoch 0: training loss: 0.5020436278184255, accuracy: 87.57916666666667
    [2026-07-22 15:42:31,105382][I][./<timed exec>:15:<module>] Epoch 0: val. loss: 0.49430568167503847, val. accuracy: 87.73333333333333
    [2026-07-22 15:42:32,654182][I][./<timed exec>:10:<module>] Epoch 1: training loss: 0.4218256519635518, accuracy: 88.97916666666667
    [2026-07-22 15:42:32,843343][I][./<timed exec>:15:<module>] Epoch 1: val. loss: 0.4125518928816978, val. accuracy: 88.81666666666666
    [2026-07-22 15:42:34,390169][I][./<timed exec>:10:<module>] Epoch 2: training loss: 0.38788456765810647, accuracy: 89.61666666666666
    [2026-07-22 15:42:34,566588][I][./<timed exec>:15:<module>] Epoch 2: val. loss: 0.3780543385668004, val. accuracy: 89.49166666666667
    [2026-07-22 15:42:36,107547][I][./<timed exec>:10:<module>] Epoch 3: training loss: 0.3679140396912893, accuracy: 90.06458333333333
    [2026-07-22 15:42:36,272014][I][./<timed exec>:15:<module>] Epoch 3: val. loss: 0.35790705363801184, val. accuracy: 89.95833333333333
    [2026-07-22 15:42:37,641550][I][./<timed exec>:10:<module>] Epoch 4: training loss: 0.35432112030188245, accuracy: 90.41458333333333
    [2026-07-22 15:42:37,808113][I][./<timed exec>:15:<module>] Epoch 4: val. loss: 0.3443210984164096, val. accuracy: 90.25833333333333
    CPU times: user 8.15 s, sys: 440 ms, total: 8.59 s
    Wall time: 8.5 s

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

    [2026-07-22 15:42:38,091946][I][ipykernel_6525/372756021:2:<module>] Test loss: 0.33227541074156763, test accuracy: 90.86

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

    [2026-07-22 15:42:38,102128][I][ipykernel_6525/2368214845:8:show_failures] Showing max 10 first failures.
    [2026-07-22 15:42:38,102586][I][ipykernel_6525/2368214845:11:show_failures] The predicted class is shown first and the correct class in parentheses.

![](index_files/figure-commonmark/cell-19-output-2.png)

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

    [2026-07-22 15:42:39,667184][I][./<timed exec>:10:<module>] Epoch 0: training loss: 0.7769493811925252, accuracy: 79.70208333333333
    [2026-07-22 15:42:39,840254][I][./<timed exec>:15:<module>] Epoch 0: val. loss: 0.7685229201266106, val. accuracy: 79.68333333333332
    [2026-07-22 15:42:41,270120][I][./<timed exec>:10:<module>] Epoch 1: training loss: 0.39049726152420045, accuracy: 89.02916666666667
    [2026-07-22 15:42:41,435632][I][./<timed exec>:15:<module>] Epoch 1: val. loss: 0.3795486579233028, val. accuracy: 89.03333333333333
    [2026-07-22 15:42:42,888811][I][./<timed exec>:10:<module>] Epoch 2: training loss: 0.3114342784086863, accuracy: 91.08333333333334
    [2026-07-22 15:42:43,067062][I][./<timed exec>:15:<module>] Epoch 2: val. loss: 0.3022882691089143, val. accuracy: 91.00833333333334
    [2026-07-22 15:42:44,583366][I][./<timed exec>:10:<module>] Epoch 3: training loss: 0.2536494802633921, accuracy: 92.7
    [2026-07-22 15:42:44,763597][I][./<timed exec>:15:<module>] Epoch 3: val. loss: 0.2454532984089344, val. accuracy: 92.60833333333333
    [2026-07-22 15:42:46,219453][I][./<timed exec>:10:<module>] Epoch 4: training loss: 0.21906284141540527, accuracy: 93.65833333333333
    [2026-07-22 15:42:46,419987][I][./<timed exec>:15:<module>] Epoch 4: val. loss: 0.21409300120269997, val. accuracy: 93.60833333333333
    CPU times: user 7.98 s, sys: 451 ms, total: 8.43 s
    Wall time: 8.23 s

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

    [2026-07-22 15:42:46,467965][I][ipykernel_6525/2368214845:8:show_failures] Showing max 10 first failures.
    [2026-07-22 15:42:46,468453][I][ipykernel_6525/2368214845:11:show_failures] The predicted class is shown first and the correct class in parentheses.

![](index_files/figure-commonmark/cell-24-output-2.png)

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

    [2026-07-22 15:42:47,608809][I][./<timed exec>:11:<module>] Epoch 1/6, Learning Rate: 0.1
    [2026-07-22 15:42:48,618980][I][./<timed exec>:16:<module>] Epoch 0: training loss: 0.39064898585279784, accuracy: 88.85416666666667
    [2026-07-22 15:42:48,922952][I][./<timed exec>:21:<module>] Epoch 0: val. loss: 0.3824256694515546, val. accuracy: 88.69166666666666
    [2026-07-22 15:42:50,552237][I][./<timed exec>:11:<module>] Epoch 2/6, Learning Rate: 0.010000000000000002
    [2026-07-22 15:42:51,491841][I][./<timed exec>:16:<module>] Epoch 1: training loss: 0.24925500146547952, accuracy: 92.58125
    [2026-07-22 15:42:51,828054][I][./<timed exec>:21:<module>] Epoch 1: val. loss: 0.23925010534127553, val. accuracy: 92.68333333333332
    [2026-07-22 15:42:53,653017][I][./<timed exec>:11:<module>] Epoch 3/6, Learning Rate: 0.010000000000000002
    [2026-07-22 15:42:54,441495][I][./<timed exec>:16:<module>] Epoch 2: training loss: 0.23246053767080108, accuracy: 93.12083333333334
    [2026-07-22 15:42:54,635711][I][./<timed exec>:21:<module>] Epoch 2: val. loss: 0.22537943425277868, val. accuracy: 93.25
    [2026-07-22 15:42:55,751275][I][./<timed exec>:11:<module>] Epoch 4/6, Learning Rate: 0.0010000000000000002
    [2026-07-22 15:42:56,992415][I][./<timed exec>:16:<module>] Epoch 3: training loss: 0.22418363230427107, accuracy: 93.37083333333334
    [2026-07-22 15:42:57,249258][I][./<timed exec>:21:<module>] Epoch 3: val. loss: 0.21702358741561573, val. accuracy: 93.425
    [2026-07-22 15:42:58,425739][I][./<timed exec>:11:<module>] Epoch 5/6, Learning Rate: 0.0010000000000000002
    [2026-07-22 15:42:59,183652][I][./<timed exec>:16:<module>] Epoch 4: training loss: 0.22311136978243787, accuracy: 93.36041666666667
    [2026-07-22 15:42:59,378944][I][./<timed exec>:21:<module>] Epoch 4: val. loss: 0.2162215127994617, val. accuracy: 93.39166666666667
    [2026-07-22 15:43:00,530541][I][./<timed exec>:11:<module>] Epoch 6/6, Learning Rate: 0.00010000000000000003
    [2026-07-22 15:43:01,300857][I][./<timed exec>:16:<module>] Epoch 5: training loss: 0.2224602335592111, accuracy: 93.3875
    [2026-07-22 15:43:01,490620][I][./<timed exec>:21:<module>] Epoch 5: val. loss: 0.21565010777612528, val. accuracy: 93.425
    CPU times: user 13.5 s, sys: 1.97 s, total: 15.4 s
    Wall time: 14.9 s
