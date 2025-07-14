# Perceptron

## 1. Single-layer perceptron

### 1.1 Origin and basic idea

* A single artificial neuron for supervised learning, originally proposed by **Frank Rosenblatt (1957)** as a binary classifier.
* Acts as a **linear classifier**: a weighted sum decides between two classes.

### 1.2 Mathematical structure

Each neuron is defined by inputs $x_i$, weights $w_i$, a **threshold** $U$, and an activation function $f(\cdot)$.

<div style="text-align:center;">
  <img src="docs/images/image.png" alt="perceptron" width="400" />
</div>

$$
a = \sum_{i=1}^{n} w_i\,x_i - U
$$

The continuous output (probability) is obtained with the **sigmoid**:

$$
y = f(a) = \sigma(a) = \frac{1}{1+e^{-a}}
$$

### 1.3 Learning algorithm (sigmoid + log-loss)

1. **Initialize** weights and threshold

   * $w_i \sim [-0.5,0.5]$
   * $U   \sim [0,1]$
2. For each example $(\mathbf{x},s)$ (label $s\in{0,1}$):

   1. Compute $a=\sum w_i x_i - U$ and $y=\sigma(a)$.
   2. **Soft error:** $\delta = y - s$.
   3. **Update** parameters (gradient descent on the log-loss):

$$
\begin{aligned}
 w_i &\leftarrow w_i - \alpha\,\delta\,x_i \\[4pt]
 U   &\leftarrow U   + \alpha\,\delta
\end{aligned}
$$

3. Repeat for several epochs until the mean loss stabilises.

### 1.4 Inference

1. Compute $a$ and $y=\sigma(a)$ on test data.
2. **Threshold** (hard decision):
   $(y\ge 0.5)\Rightarrow 1$; $(y<0.5)\Rightarrow 0$.

---

## 2. Multilayer perceptron (MLP)

### 2.1 Motivation

Hidden layers allow the network to learn **non-linear** decision boundaries.

### 2.2 Architecture

<img src="docs/images/image-5.png" alt="mlp" width="460" />

### 2.3 Learning via back-propagation

For each weight $w_{ij}$

$w_{ij} \leftarrow w_{ij} - \alpha,\dfrac{\partial E}{\partial w_{ij}}$

Requires differentiable activations (sigmoid, ReLU, tanh, …).

---

## 3. Activation functions

| Type         | Formula                                   | Graphic                                                         |
| ------------ | ----------------------------------------- | --------------------------------------------------------------- |
| **Linear**   | $f(a)=a$                                | —                                                               |
| **Sigmoid**  | $\displaystyle f(a)=\frac{1}{1+e^{-a}}$ | <img src="docs/images/image-2.png" alt="sigmoid" width="190" /> |
| **Gaussian** | selective for intermediate values         | <img src="docs/images/image-1.png" alt="gauss" width="190" />   |
| **ReLU**     | $f(a)=\max(0,a)$                        | —                                                               |

---

## 4. Quick formula recap

* **Aggregation:** $a=\sum w_i x_i - U$
* **Continuous output:** $y=\sigma(a)$
* **Soft error:** $\delta = y - s$
* **Sigmoid/log-loss update:**
  $w_i \leftarrow w_i - \alpha,\delta,x_i$,
  $U \leftarrow U + \alpha,\delta$
* **MLP gradient step:** $w \leftarrow w - \alpha \nabla E$
