---
title: "Deep RL 2 Imitation Learning"
date: 2021-04-06
categories: DeepRL
---
The framework of imitation learning tackles reinforcement learning as a supervised learning problem.

Some Basic Notations and TerminologiesPermalink
======
$$\begin{eqnarray*}
&&o_t: \text{observation at time }t \\
&&s_t: \text{state at time }t \\
&&a_t: \text{action at time }t \\
&&\pi(a_t\mid o_t): \text{policy, a distribution over actions }a_t\text{ given observation }o_t \\
&&p(s_{t+1}\mid s_t, a_t): \text{transition probabilities or dynamics or model} \\
&&p(o_t\mid s_t, a_t): \text{conditional observation probabilities}
\end{eqnarray*}$$

**More on policy $$\pi$$**

When  $$\pi(a_t\mid o_t)$$ is a delta function, it’s a deterministic mapping from observation space to action space. In many cases we want to parameterize $$\pi$$ by a neural net, and optimize the some objective to improve the policy. For example, when the action space is discrete, we can let $$\pi_{\theta}(a_t\mid o_t)= \text{Cat}(\alpha_{\theta}(o_t))$$ and the neural net produces probabilities/logits of each category $$(\alpha_{\theta}(o_t))$$; when the action space is continuous,  we can let $$\pi_{\theta}(a_t\mid o_t) = \mathcal{N}(\mu_{\theta}(o_t),\Sigma_{\theta}(o_t))$$ and the neural network produces mean and variance of the Gaussian distribution.

Sometimes we use  instead, which is a more restrictive special case, comparing to . Because state is usually assumed to be Markovian, i.e. , but we don’t pose such assumption on .

An illustrative example os $$o_t$$ and $$s_t$$ (Figures are all from [CS285](http://rail.eecs.berkeley.edu/deeprlcourse/) unless otherwise pointed out.):
<div align='center'><img src="../assets/images/285-2-os.png" alt="o and s" width="700"></div>

A graphical model that shows the causal relationship of different quantities:
<div align='center'><img src="../assets/images/285-2-markov.png" alt="markov" width="700"></div>


Suppose we want to train an agent to drive a car autonomously, in imitation learning framework, we first collect data $$\{(o_t,a_t)\}$$ from human drivers, where ’s are the images from the car’s camera and ’s are the driver’s behaviors. We then parameterize policy $$\pi_{\theta}(a_t\mid o_t)$$ by a neural network which takes in observation $$o_t$$ and output action $$\hat{a}_t$$. We want the neural net policy to act as close to human as possible, and therefore optimize the following objective:

$$\begin{equation}\label{eq:orig}
\ell = \sum_{t}\mathcal{L}(\hat{a}_t, a_t)
\end{equation}$$
 
where $$\mathcal{L}$$ can be mean square error (continuous action space), cross entropy (discrete action space) etc.

BUT $$\ref{eq:orig}$$ might not work