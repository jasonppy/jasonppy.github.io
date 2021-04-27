---
title: "Deep RL 4 Policy Gradient"
date: 2021-04-13
categories:
  - DeepRL
tags:
  - Notes
  - RL
---
In this lecture, we will study the classic policy gradient methods, which includes the REINFORCE algorithm, off-policy policy gradient method, and several common tricks for making these methods to work better.

## 1 REINFORCE
Policy gradient methods are one of the most straightforward methods of RL because they directly optimizes the goal $$J(\theta)$$ of RL by gradient descent (well, it's actually gradient ascent, because we are maximizing the objective, but this doesn't make a difference because we can also add a minus sign in the code).

$$\begin{equation}\label{first}J(\theta) = \mathbb{E}_{p_{\theta}(\tau)}\sum_{t=1}^T r(s_t, a_t)\end{equation}$$

Where $$\tau = (s_1, a_1, \cdots, s_T, a_t)$$ and $$p_{\theta}(\tau) = p(s_1)\prod_{t=1}^{T}p(s_{t+1}\mid s_t, a_t)\pi_{\theta}(a_t\mid s_t)$$

Now let's derive the REINFORCE algorithm, which is the most basic policy gradient method. We simply take the derivative of $$J(\theta)$$:

$$\begin{align}
\nabla_{\theta}J(\theta) &= \nabla_{\theta}\mathbb{E}_{p_{\theta}(\tau)}\sum_{t=1}^T r(s_t, a_t) \nonumber \\
&= \nabla_{\theta}\int_{\tau} p_{\theta}(\tau)\left(\sum_{t=1}^T r(s_t, a_t)\right) \text{d}\tau \nonumber \\
&= \int_{\tau} \nabla_{\theta}p_{\theta}(\tau)\left(\sum_{t=1}^T r(s_t, a_t)\right) \text{d}\tau \label{equality1} \\
&= \int_{\tau} p_{\theta}(\tau)\nabla_{\theta}\log p_{\theta}(\tau)\left(\sum_{t=1}^T r(s_t, a_t)\right) \text{d}\tau \label{equality2} \\
&= \mathbb{E}_{p_{\theta}(\tau)}\nabla_{\theta}\log p_{\theta}(\tau)\left(\sum_{t=1}^T r(s_t, a_t)\right) \label{inter} \\
\end{align}$$

To get from equation $$\ref{equality1}$$ to equation $${\ref{equality2}}$$, we used the equality $$\nabla_{\theta}\log p_{\theta}(\tau) = \frac{\nabla_{\theta}p_{\theta}(\tau)}{p_{\theta}(\tau)}$$

The result is ideal in that the derivative can still be writen as an expectation, which means we can obtain the Monte Carlo estimate of it using samples. However, the term $$p_{\theta}(\tau)$$ is not known as we don't know the model $$p(s_{t+1}\mid s_t, a_t)$$. But this is actually not a problem, if we note

$$\begin{align}
\nabla_{\theta}\log p_{\theta}(\tau) 
&= \nabla_{\theta}\log p(s_1)\prod_{t=1}^{T}p(s_{t+1}\mid s_t, a_t)\pi_{\theta}(a_t\mid s_t) \label{orig_1} \\
&= \enclose{downdiagonalstrike}{\nabla_{\theta}\left[\log \left(p(s_1)\prod_{t=1}^{T} p(s_{t+1}\mid s_t, a_t) \right)\right]} + \nabla_{\theta}\sum_{t=1}^{T}\log \pi_{\theta}(a_t\mid s_t) \label{orig_2} \\
&=  \sum_{t=1}^{T}\nabla_{\theta}\log \pi_{\theta}(a_t\mid s_t)  \label{orig_3}
\end{align}$$

plug this term into the original result equation $$\ref{inter}$$, we have

$$\begin{align*} \label{gradient}
\nabla_{\theta}J(\theta) = \mathbb{E}_{p_{\theta}(\tau)} \left(\sum_{t=1}^{T}\nabla_{\theta}\log \pi_{\theta}(a_t\mid s_t)\right)\left(\sum_{t=1}^T r(s_t, a_t)\right)
\end{align*}$$

In practice, after running the agent in environment or simulator, we have trajectories $$\{ \tau^i \}_{i=1}^{N}$$ where $$\tau^i = (s^i_1, a^i_1, \cdots, s^i_T, a^i_T)$$ and rewards $$\{ r^i_t \}_{i=1, t=1}^{N,T}$$, then we can obtain Monte Carlo estimate of the gradient:

$$\begin{align} \label{gradient_estimate}
\nabla_{\theta}J(\theta) = \frac1N \sum_{i=1}^{N}\left(\sum_{t=1}^{T}\nabla_{\theta}\log \pi_{\theta}(a^i_t\mid s^i_t)\right)\left(\sum_{t=1}^T s^i_t\right)
\end{align}$$

For a more rigorous notation, the left hand side of equation $$\ref{gradient_estimate}$$ should be $$\widehat{\nabla_{\theta}J(\theta)}$$, because it's an estimator of the original quantity. However, since the objectives and gradients are always estimated using samples in Deep RL algorithms, we slightly abuse the notation i.e. do not use the 'hat' when the quantity is an estimator.

We can also get Monte Carlo estimate of the expected reward of the current policy using rewards $$\{ r^i_t \}_{i=1, t=1}^{N,T}$$:

$$\begin{align} \label{obj_estimate}
J(\theta) = \frac1N \sum_{i=1}^N \sum_{t=1}^T r^i_t
\end{align}$$

This means we can evaluate policy easily when using policy gradient methods.

The following is the REINFORCE algorithm:

1. initialize policy $$\pi_{\theta}$$
2. run policy $$\pi_{\theta}$$ to generate sample trajectories $$\{ \tau^i \}_{i=1}^{N}$$
3. estimate gradient: $$\nabla_{\theta}J(\theta) = \frac1N \sum_{i=1}^{N}\left(\sum_{t=1}^{T}\nabla_{\theta}\log \pi_{\theta}(a^i_t\mid s^i_t)\right)\left(\sum_{t=1}^T s^i_t\right)$$
3. update policy: $$\theta \leftarrow \theta + \alpha\nabla_{\theta}J(\theta)$$. Go to 2.

Finally I want to point out that the REINFORCE algorithm also works in POMDP, where we don't know state $$s_t$$, but can only observe observation $$o_t$$, which means the policy is $$\pi_{\theta}(a_t\mid o_t)$$. This is clear if we write out the trajectory distribution in POMDP:

$$\begin{equation}\label{pomdp}p_{\theta}(\tau) = p(s_1)\prod_{i=1}^Tp(s_{t+1}\mid s_t, a_t)\pi_{\theta}(a_t\mid o_t)p(o_t\mid s_t)\end{equation}$$

where $$\tau = (s_1, o_1, a_1, s_2, o_2, a_2,\cdots, s_T, o_T, a_t)$$. Note that we don't attempt to learn the emission probability distribution $$p(o_t\mid s_t)$$. If we plug in equation $$\ref{pomdp}$$ in to the derivation of original policy gradient equation $$\ref{orig_1}$$, all the terms except for the one contains $$\theta$$ will be zero when we take the derivative at equation $$\ref{orig_2}$$. So the policy gradient for POMDP is 

$$\begin{equation} \nabla_{\theta}J(\theta) = \frac1N\sum_{i=1}^N\sum_{t=1}^T\nabla_{\theta}\log \pi_{\theta}(a_t^i\mid o_t^i)s^i_t\end{equation}$$.

The REINFORCE algorithm is very straigtforward and match our intuition about common deep learning paradigm (i.e. using gradient descent to optimize our goal). But it has two major issues: first, the training is not very stable as the gradient estimator as large variance; second, the algorithm is not very efficient as new trajectories need to be collected after each gradient update. We will next introduce ways to get around with these two issues. At the end of this lecture, we will introduce how policy gradient method can be implemented in common automatic differentiation packages like TensorFlow and PyTorch.
## 2 Variance Reduction
### 2.1 Causality
We change the objective using a very simple common sense on causality --- future states and actions cannot affect past rewards. Let's see how causality can help us to derive a new objective and gradient with lower variance. 

We first write down the original objective:

$$J(\theta) = \mathbb{E}_{p_{\theta}(\tau)}\sum_{t=1}^T r(s_t, a_t)$$

Switch summation and expectation:
$$\begin{equation}\label{orig_obj}J(\theta) = \sum_{t=1}^{T}\mathbb{E}_{p_{\theta}(\tau)}r(s_t, a_t)\end{equation}$$

Denote $$\tau_t = (s_1, a_1, \cdots, s_t, a_t)$$, since future states and actions cannot affect past rewards, therefore optimizing future actions cannot change improve past rewards. So, let's take out the future states and actions from the objective:

$$\begin{equation}\label{inter1}J(\theta) = \sum_{t=1}^{T}\mathbb{E}_{p_{\theta}(\tau_t)}r(s_t, a_t)\end{equation}$$

Go through very similar steps as before to differentiate $$J(\theta)$$ we have

$$\begin{equation}\label{diff_J} \nabla_{\theta}J(\theta) = \sum_{t=1}^{T}\mathbb{E}_{p_{\theta}(\tau_t)}\nabla_{\theta}\log p_{\theta}(\tau_t) r(s_t, a_t)\end{equation}$$

Still very similar as previous steps:
$$\begin{align}
\nabla_{\theta}\log p_{\theta}(\tau_t) 
&= \nabla_{\theta}\log p(s_1)\pi_{\theta}(a_1\mid s_1)\prod_{t'=2}^{t}p(s_{t'}\mid s_{t'-1}, a_{t'-1})\pi_{\theta}(a_{t'}\mid s_{t'}) \nonumber \\
&= \enclose{downdiagonalstrike}{\nabla_{\theta}\left[\log \left(p(s_1)\prod_{t'=2}^{t} p(s_{t'}\mid s_{t'-1}, a_{t'-1}) \right)\right]} + \nabla_{\theta}\sum_{t'=1}^{t}\log \pi_{\theta}(a_{t'}\mid s_{t'}) \nonumber\\
&= \sum_{t'=1}^{t}\nabla_{\theta}\log \pi_{\theta}(a_{t'}\mid s_{t'}) \label{cancel}
\end{align}$$

Plug equation $$\ref{cancel}$$ into $$\ref{diff_J}$$, we have

$$\begin{align} \nabla_{\theta}J(\theta) = \sum_{t=1}^{T}\mathbb{E}_{p_{\theta}(\tau_t)}\sum_{t'=1}^{t}\nabla_{\theta}\log \pi_{\theta}(a_{t'}\mid s_{t'}) r(s_t, a_t)\end{align}$$

and this can be approximated by sample trajectories:
$$\begin{align} \nabla_{\theta}J(\theta) 
&= \sum_{t=1}^{T} \frac1N \sum_{i=1}^{N}\sum_{t'=1}^{t}\nabla_{\theta}\log \pi_{\theta}(a_{t'}^i\mid s_{t'}^i) r_t^i \label{1}\\
&= \frac1N \sum_{i=1}^{N}\sum_{t=1}^{T}\sum_{t'=1}^{t}\nabla_{\theta}\log \pi_{\theta}(a_{t'}^i\mid s_{t'}^i) r_t^i \label{2}\\
&= \frac1N \sum_{i=1}^{N}\sum_{t'=1}^{T}\sum_{t=t'}^{T}\nabla_{\theta}\log \pi_{\theta}(a_{t'}^i\mid s_{t'}^i) r_t^i \label{3}\\
&= \frac1N \sum_{i=1}^{N}\sum_{t=1}^{T}\sum_{t'=t}^{T}\nabla_{\theta}\log \pi_{\theta}(a_{t}^i\mid s_{t}^i) r_{t'}^i \label{4}\\
&= \frac1N \sum_{i=1}^{N}\sum_{t=1}^{T}\nabla_{\theta}\log \pi_{\theta}(a_{t}^i\mid s_{t}^i) \sum_{t'=t}^{T}r_{t'}^i \label{5}\\
\end{align}$$

The above equations seems scary, but they are exchange of summation order (equation $$\ref{1}$$, $$\ref{2}$$, $$\ref{3}$$) and change of notations (equation $$\ref{4}$$). Now let's recall the gradient that doesn't incorporate causality:

$$\begin{equation}\label{orig_gradient}\frac1N \sum_{i=1}^{N}\sum_{t=1}^{T}\nabla_{\theta}\log \pi_{\theta}(a_{t}^i\mid s_{t}^i) \sum_{t'=1}^{T}r_{t'}^i\end{equation}$$

Note that gradient that incorporates causality (equation $$\ref{5}$$) are really the same as the gradient that doesn't incorporate causality (equation $$\ref{orig_gradient}$$), but with past rewards in the reward summation subtracted. This gives less terms in the summation, and therefore gives smaller variance.

### 2.2 Baselines
We can also reduce the variance of by introducing a baseline.

To simplify the notation, let's use $$r(\tau)$$ to denote $$\sum_{t=1}^{T}r(s_t, a_t)$$. We want to use Monte Carlo estimate to approximate the gradient

$$\begin{equation}\label{orig_grad}\nabla_{\theta}J(\theta) = \mathbb{E}_{p_{\theta}(\tau)}\nabla_{\theta}\log p_{\theta}(\tau)r(\tau)\end{equation}$$

To introduce baseline, we simply change the gradient to be

$$\begin{equation}\label{base_grad}\nabla_{\theta}J(\theta) = \mathbb{E}_{p_{\theta}(\tau)}\nabla_{\theta}\log p_{\theta}(\tau)\left(r(\tau)-b\right)\end{equation}$$

Where $$b$$ is the baseline, which is not a function of $$\tau$$. 

You might ask: 1. Is the gradient with the basedline added the same as the original gradient? 2. will the baseline lead to a lower variance and if so, what $$b$$ gives lowest variance?

The answer to the first question is yes. To see that, we just need a bit of calculation:

$$\begin{align*} \mathbb{E}_{p_{\theta}(\tau)}\nabla_{\theta}\log p_{\theta}(\tau)b 
&= b\mathbb{E}_{p_{\theta}(\tau)}\nabla_{\theta}\log p_{\theta}(\tau) \\
&= b\int_{\tau}p_{\theta}(\tau)\nabla_{\theta}\log p_{\theta}(\tau)\text{d}\tau \\
&= b\int_{\tau}p_{\theta}(\tau)\frac{\nabla_{\theta}p_{\theta}(\tau)}{p_{\theta}(\tau)} \text{d}\tau\\
&= b\nabla_{\theta}\int_{\tau}p_{\theta}(\tau) \\
&= 0
\end{align*}$$

To answer the second question, we need to do a bit more calculation. Let's first calculate the variance!

With the variance formula $$\mathbb{V}(x) = \mathbb{E}x^2 + (\mathbb{E}x)^2$$, we have

$$\begin{equation} 
\mathbb{V}\left[\nabla_{\theta}\log p_{\theta}(\tau)(r(\tau)-b)\right] = \mathbb{E}(\nabla_{\theta}\log p_{\theta}(\tau)(r(\tau)-b))^2 + (\mathbb{E}\nabla_{\theta}\log p_{\theta}(\tau)(r(\tau)-b))^2
\end{equation}$$

Let's minimize this term, note that it's a quadratic function of $$b$$ and therefore to minimize it w.r.t $$b$$, we just need to find the stationary point. Also note that the second term is just $$(\mathbb{E}\nabla_{\theta}\log p_{\theta}(\tau)r(\tau))^2$$, so when we take the derivative w.r.t $$b$$ this term will be $$0$$. The derivative is:

$$\begin{align*} 
\nabla_{b}\mathbb{V}\left[\nabla_{\theta}\log p_{\theta}(\tau)(r(\tau)-b)\right]
&= \nabla_{\theta}\mathbb{E}\nabla_{\theta}(\log p_{\theta}(\tau))^2(r(\tau)^2-2br(\tau) + b^2) \\
&= -2\mathbb{E}\nabla_{\theta}(\log p_{\theta}(\tau))^2r(\tau) + 2b\mathbb{E}\nabla_{\theta}(\log p_{\theta}(\tau))^2
\end{align*}$$

set it to be $$0$$, we have

$$\begin{equation}
b^* = \frac{\mathbb{E}\nabla_{\theta}(\log p_{\theta}(\tau))^2r(\tau)}{\mathbb{E}\nabla_{\theta}(\log p_{\theta}(\tau))^2}
\end{equation}$$

$$b^*$$ can be interpreted as the expected reward weighted by gradient. When set $$b$$ to be $$0$$, we recover the original gradient. In RL, people usually just use unweighted expected reward, i.e. $$b=\mathbb{E}r(\tau)$$. So during training, the gradient with baseline is 

$$\begin{equation}
\nabla_{\theta}j(\theta) = \frac1N \sum_{i=1}^N\nabla_{\theta}p_{\theta}(\tau^i) (r(\tau^i) - b)
\end{equation}$$

where $$b = \frac1N \sum_{i=1}^Nr(\tau^i)$$. Note that we actually don't know if this baseline can reduce variance as it is probably not $$ b^* $$, but in practice people find it can stablize the training.

<p> </p>

The use causality and baseline are the two most common techniques to reduce the variance policy gradient. However, there are other techniques, in the next lecture, we will introduce actor-critic algorithm, which can be view as a low variance variant of policy gradient method.

## 3 Off-policy Policy Gradient
In this section, we study how to make policy gradient more efficient, and the idea is to make the algorithm from on-policy to off-policy. 

We will start from the original policy gradient and use the idea of important sampling to make the expectation to be no longer over the current policy. Sepcifically, we want the expectation to be over some old policy $$\pi_{\theta}$$ i.e. $$\pi_{\theta}$$ is many gradient steps before the current policy $$\pi_{\theta'}$$.

$$\begin{align}
\nabla_{\theta'}J(\theta')
&= \mathbb{E}_{\tau \sim p_{\theta'}(\tau)}\sum_{t=1}^T\nabla_{\theta'}\log \pi_{\theta'}(a_t\mid s_t) r(s_t, a_t) \label{start}\\
&= \sum_{t=1}^{T}\mathbb{E}_{s_t, a_t \sim p_{\theta'}(s_t, a_t)}\nabla_{\theta'}\log \pi_{\theta'}(a_t\mid s_t) r(s_t, a_t) \label{switch} \\
&= \sum_{t=1}^{T}\int_{s_t, a_t}p_{\theta'}(s_t, a_t)\nabla_{\theta'}\log \pi_{\theta'}(a_t\mid s_t) r(s_t, a_t) \text{d}(s_t, a_t) \label{integral}\\
&= \sum_{t=1}^{T}\int_{s_t, a_t}p_{\theta'}(s_t, a_t)\frac{p_{\theta}(s_t, a_t)}{p_{\theta}(s_t, a_t)}\nabla_{\theta'}\log \pi_{\theta'}(a_t\mid s_t) r(s_t, a_t) \text{d}(s_t, a_t) \\
&= \sum_{t=1}^{T}\mathbb{E}_{s_t, a_t \sim p_{\theta}(s_t, a_t)}\frac{p_{\theta'}(s_t, a_t)}{p_{\theta}(s_t, a_t)}\nabla_{\theta'}\log \pi_{\theta'}(a_t\mid s_t) r(s_t, a_t) \label{importance_sampling} \\
&= \sum_{t=1}^{T}\mathbb{E}_{s_t, a_t \sim p_{\theta'}(s_t, a_t)}\frac{p_{\theta'}(s_t)\pi_{\theta'}(a_t\mid s_t)}{p_{\theta}(s_t)\pi_{\theta}(a_t\mid s_t)}\nabla_{\theta'}\log \pi_{\theta'}(a_t\mid s_t) r(s_t, a_t) \label{marginal}\\
&\approx \sum_{t=1}^{T}\mathbb{E}_{s_t, a_t \sim p_{\theta}(s_t, a_t)}\enclose{downdiagonalstrike}{\frac{p_{\theta'}(s_t)}{p_{\theta}(s_t)}}\frac{\pi_{\theta'}(a_t\mid s_t)}{\pi_{\theta}(a_t\mid s_t)}\nabla_{\theta'}\log \pi_{\theta'}(a_t\mid s_t) r(s_t, a_t) \label{cross_out}\\
&= \sum_{t=1}^{T}\mathbb{E}_{s_t, a_t \sim p_{\theta}(s_t, a_t)}\frac{\pi_{\theta'}(a_t\mid s_t)}{\pi_{\theta}(a_t\mid s_t)}\nabla_{\theta'}\log \pi_{\theta'}(a_t\mid s_t) r(s_t, a_t) \label{res}\\
\end{align}$$


From equation $$\ref{start} \text{ to } \ref{switch}$$, we switch the summation and expectation sign and also set the expectation to be over marginal distribution of $$(s_t, a_t)$$ rather than the whole trajectory. From $$\ref{switch}$$ to $$\ref{importance_sampling}$$, we use importance sampling to change the underlying distribution of the expectation, where $$\theta$$ is the parameter of some old policy (e.g. the policy that is many gradient steps before the current policy). This step makes the gradient off-policy, as samples are from the old policy rather than the current policy. Note that the distribution $$p_{\theta}(s_t, a_t)$$ or $$p_{\theta'}(s_t, a_t)$$ is unknown if we don't know the transition model, therefore we write them as a product of state marginal $$p_{\theta'}(s_t)$$ and policy $$\pi_{\theta'}(a_t\mid s_t)$$ in $$\ref{marginal}$$, and then cross out the state marginal in $$\ref{cross_out}$$. Just crossing out the term will lead to an systematic estimation error on the gradient estimation, but we will see in later lecture when we introduce advanced policy gradient methods that the error is bounded when the gap between $$\pi_{\theta}$$ and $$\pi_{\theta'}$$ are not too big.

Finally we have equation $$\ref{res}$$ which is off-policy policy gradient. An intuitive explanation of this is that the off-policy policy gradient is on-policy  policy gradient but each term is weighted by $$\frac{\pi_{\theta'}(a_t\mid s_t)}{\pi_{\theta}(a_t\mid s_t)}$$. 

The Monte Carlo estimate of the off-policy policy gradient is 

$$\begin{equation}
\nabla_{\theta'}J(\theta') = \frac1N \sum_{i=1}^{N}\sum_{t=1}^{T}\frac{\pi_{\theta'}(a^i_t\mid s^i_t)}{\pi_{\theta}(a^i_t\mid s^i_t)}\nabla_{\theta'}\log \pi_{\theta'}(a^i_t\mid s^i_t) r^i_t 
\end{equation}$$

Where the trajectories are sampled from old policy $$p_{\theta}(\tau)$$.

We can also use the log derivative trick to write the off-policy policy gradient as

$$\begin{equation}\label{off_use}
\nabla_{\theta'}J(\theta') = \frac1N \sum_{i=1}^{N}\sum_{t=1}^{T}\frac{\nabla_{\theta'}\pi_{\theta'}(a^i_t\mid s^i_t)}{\pi_{\theta}(a^i_t\mid s^i_t)}r^i_t 
\end{equation}$$

This will be useful when we implement the algorithm (see next section). 

## 4 A Note on Implementing Policy Gradient
Policy gradient using gradient ascent to optimize the objective 

$$J(\theta) = \frac1N \sum_{i=1}^{N}\sum_{t=1}^Tr^i_t$$

Since we parameterize policy $$\pi_{\theta}$$ using a neural netowrk, we will use an automatic differentiation package like TensorFlow or PyTorch to calculate the gradient and update weights. However, autodiff package will be default not get the policy gradient that we derived but try to backprop through samples, which will not work as we don't know the trajectory distribution and the reward function. How do we specify the gradient we want autodiff package to use when it updates weights?

One solution is to derive the gradient manually and explicitly write it in our code. But this can be very tedious and essentially bring us to pre-autodiff era.

Luckily, we use a pseudo-objective to trick autodiff package to derive the policy gradient during backprop and use it to update weights. Recall that the policy gradient is 

$$\begin{equation*}
\nabla_{\theta}J(\theta) = \frac1N \sum_{i=1}^{N}\sum_{t=1}^{T}\nabla_{\theta}\log \pi_{\theta}(a^i_t\mid s^i_t) r^i_t
\end{equation*}$$

The objective we provide to autodiff packages is 

$$\begin{equation*}
J(\theta) = \frac1N \sum_{i=1}^{N}\sum_{t=1}^{T}\log \pi_{\theta}(a^i_t\mid s^i_t) r^i_t
\end{equation*}$$

The the derivative of it is exactly the policy gradient.

For off-policy policy gradient, similarly, to get gradient $$\ref{off_use}$$, the pseudo-objective is we provide in the code is 

$$\begin{equation*}
\nabla_{\theta}J(\theta) = \frac1N \sum_{i=1}^{N}\sum_{t=1}^{T}\frac{\pi_{\theta}(a^i_t\mid s^i_t)}{\pi_{\theta'}(a^i_t\mid s^i_t)}r^i_t 
\end{equation*}$$


<br/><br/>
Lastly, here are some recommended papers on policy gradient methods

* Classic papers
  * Williams (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning  \[[paper](https://people.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf)\] *(this introduces the REINFORCE algorithm)*
  * Baxter & Bartlett (2001). Infinite-horizon policy-gradient estimation: temporally decomposed policy gradient \[[paper](https://arxiv.org/pdf/1106.0665.pdf)\]  *(not the first paper on this! see actor-critic section later)*
  * Peters & Schaal (2008). Reinforcement learning of motor skills with policy gradients \[[paper](http://www-clmc.usc.edu/publications/P/peters-NN2008.pdf)\] *(very accessible overview of optimal baselines and natural gradient)*

* Deep reinforcement learning policy gradient papers
  * Levine & Koltun (2013). Guided policy search: deep RL with importance sampled policy
gradient \[[paper](https://graphics.stanford.edu/projects/gpspaper/gps_full.pdf)\] *(unrelated to later discussion of guided policy search)*
  * Schulman, L., Moritz, Jordan, Abbeel (2015). Trust region policy optimization \[[paper](https://arxiv.org/pdf/1502.05477.pdf)\] *(deep RL with natural policy gradient and adaptive step size)*
  * Schulman, Wolski, Dhariwal, Radford, Klimov (2017). Proximal policy optimization algorithms \[[paper](https://arxiv.org/pdf/1707.06347.pdf)\] *(deep RL with importance sampled policy gradient)*