# Variational autoencoder

---

## What is variational autoencoder?

* Structure: Autoencoder


* Objective function: Variational inference

---

### Structure: Autoencoder

- use MLP as encoder and decoder.
- The objective of common version autoencoder is $\mathcal{L}(X, X')=||X-X'||_2^2$, where $X$ is the original image and $X'$ is reconstruction image.

---

### Objective function: Variational inference

- $\max \ln p(\mathbf{X})=\int q(\mathbf{Z})\ln(\frac{p(\mathbf{X,Z})}{q(\mathbf{Z})})d\mathbf{Z}-\int q(\mathbf{Z})\ln(\frac{p(\mathbf{Z}|\mathbf{X})}{q(\mathbf{Z})})d\mathbf{Z}$

  $=\mathcal{L}(q)+KL(q||p)$

- The objective of variational autoencoder is to comfirm that the posterior probability density function in the encoder approximates the posterior probability density function in the decoder. 

- The final objective function of variational autoencoder is$$\max\limits_{\phi,\theta}\mathcal{L}(\theta,\phi,x^{(i)})=-KL(q_\phi(\mathbf{z}|\mathbf{x}^{(i)})||p_\theta(\mathbf{z}))+E_{q_\phi(z|x^{(i)})}[\log p_\theta(\mathbf{x}^{(i)}|\mathbf{z})]$$. 


---

## How to optimize the model?

Objective:

$$\max\limits_{\phi,\theta}\mathcal{L}(\theta,\phi,x^{(i)})=-KL(q_\phi(\mathbf{z}|\mathbf{x}^{(i)})||p_\theta(\mathbf{z}))+E_{q_\phi(z|x^{(i)})}[\log p_\theta(\mathbf{x}^{(i)}|\mathbf{z})]$$. 

* The SGVB estimator and AEVB algorithm


* Reparameterization trick

---

### The SGVB estimator and AEVB algorithm

- Objective: 

  estimate the lower bound and its derivatives w.r.t. the parameters

- Method: 

  $${\mathcal{L}}(\theta, \phi;x^{i})\simeq\tilde{\mathcal{L}}(\theta, \phi;x^{i}) = -KL(q_\phi(z|x)||p_{\theta}(z)) + \frac{1}{L}\sum_{l=1}^L\ln p_\theta(x^{(i)}|z^{(i,l)}) $$

  ​

  Why? 

  1. KL-divergence can often be integrated analytically. 

     When both prior $p_\theta(\mathbf{z})=\mathcal{N}(0, \mathbf{I})$ and posterior approximation $q_\phi(\mathbf{z}|\mathbf{x}^{(i)})$ are Gaussion, 

     $-KL(q_\phi(z|x)||p_{\theta}(z)) =\frac{1}{2}\sum_{j=1}^J(1+\log(\sigma_j^2-\mu_j^2-\sigma_j^2))$

  2. This estimator has less variance. 

---

### Reparameterization trick

- Objective: 

  generate the samples from $q_\theta(\mathbf{z}|\mathbf{x})$, rewrite an exception w.r.t. $q_\phi(\mathbf{z}|\mathbf{x})$ such that the Monte Carlo estimator of the expectation is differentiable w.r.t. $\phi$.

- Method:

  $\epsilon^{(l)}\sim p(\epsilon)$ , $\mathbf{z}=g_\phi(\epsilon, \mathbf{x})$

  example: Let $z\sim p(z|x)=\mathcal{N}(\mu,\sigma^2)$, then a reparameterization is $z=\mu+\sigma\epsilon$, where $\epsilon$ is an auxiliary noise variable $\epsilon\sim\mathcal{N}(0,1) $. 

---

## Algorithm

Input: 