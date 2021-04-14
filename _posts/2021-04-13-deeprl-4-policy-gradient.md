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

$$J(\theta) = \mathbb{E}_{p_{\theta}(\tau)}\sum_{t=1}^T r(s_t, a_t)$$

Where $$\tau = (s_1, a_1, \cdots, s_T, a_t)$$ and $$p_{\theta}(\tau) = p(s_1)\prod_{t=1}^{T}p(s_{t+1}\mid s_t, a_t)\pi_{\theta}(a_t\mid s_t)$$

Now let's derive the REINFORCE algorithm, which is the most basic policy gradient method. We simply take the derivative of $$J(\theta)$$:

$$\begin{align}
\nabla_{\theta}J(\theta) &= \nabla_{\theta}\mathbb{E}_{p_{\theta}(\tau)}\sum_{t=1}^T r(s_t, a_t) \nonumber \\
&= \nabla_{\theta}\int_{\tau} p_{\theta}(\tau)\left(\sum_{t=1}^T r(s_t, a_t)\right) d\tau \nonumber \\
&= \int_{\tau} \nabla_{\theta}p_{\theta}(\tau)\left(\sum_{t=1}^T r(s_t, a_t)\right) d\tau \label{equality1} \\
&= \int_{\tau} p_{\theta}(\tau)\nabla_{\theta}\log p_{\theta}(\tau)\left(\sum_{t=1}^T r(s_t, a_t)\right) d\tau \label{equality2} \\
&= \mathbb{E}_{p_{\theta}(\tau)}\nabla_{\theta}\log p_{\theta}(\tau)\left(\sum_{t=1}^T r(s_t, a_t)\right) \label{inter} \\
\end{align}$$

To get from equation $$\ref{equality1}$$ to equation $${\ref{equality2}}$$, we used the equality $$\nabla_{\theta}\log p_{\theta}(\tau) = \frac{\nabla_{\theta}p_{\theta}(\tau)}{p_{\theta}(\tau)}$$

The result is ideal in that the derivative can still be writen as an expectation, which means we can get an unbiased estimation of it using samples. However, the term $$p_{\theta}(\tau)$$ is not known as we don't know the model $$p(s_{t+1}\mid s_t, a_t)$$. But this is actually not a problem, if we note

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

In practice, after running the agent in environment or simulator, we have trajectories $$\{ \tau^i \}_{i=1}^{N}$$ where $$\tau^i = (s^i_1, a^i_1, \cdots, s^i_T, a^i_T)$$ and rewards $$\{ r^i_t \}_{i=1, t=1}^{N,T}$$, then we can obtain an unbiased estimation of the gradient:

$$\begin{align} \label{gradient_estimate}
\nabla_{\theta}J(\theta) = \frac1N \sum_{i=1}^{N}\left(\sum_{t=1}^{T}\nabla_{\theta}\log \pi_{\theta}(a^i_t\mid s^i_t)\right)\left(\sum_{t=1}^T s^i_t\right)
\end{align}$$

For a more rigorous notation, the left hand side of equation $$\ref{gradient_estimate}$$ should be $$\widehat{\nabla_{\theta}J(\theta)}$$, because it's an estimator of the original quantity. However, since the objectives and gradients are always estimated using samples in Deep RL algorithms, we slightly abuse the notation i.e. do not use the 'hat' when the quantity is an estimator.

We can also get an unbiased estimate of the expected reward of the current policy using rewards $$\{ r^i_t \}_{i=1, t=1}^{N,T}$$:

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

## 2 Variance Reduction
It turns out that the original policy gradient suffers from high variance, which makes the training very unstable. In this section, we introduce several variance reduction techniques. Some of them are very widely used even beyond RL.

### 2.1 Causality
We change the objective using a very simple common sense on causality --- future states and actions cannot affect past rewards. Let's see how causality can help us to derive a new objective and gradient with lower variance. 

We first write down the original objective:
$$J(\theta) = \mathbb{E}_{p_{\theta}(\tau)}\sum_{t=1}^T r(s_t, a_t)$$

Switch summation and expectation:
$$\begin{equation}\label{orig_obj}J(\theta) = \sum_{t=1}^{T}\mathbb{E}_{p_{\theta}(\tau)}r(s_t, a_t)\end{equation}$$

Denote $$\tau_t = (s_1, a_1, \cdots, s_t, a_t)$$, since future states and actions cannot affect past rewards, therefore optimizing future actions cannot change improve past rewards. So, let's take out the future states and actions from the objective:
$$\begin{equation}\label{inter1}J(\theta) = \sum_{t=1}^{T}\mathbb{E}_{p_{\theta}(\tau_t)}r(s_t, a_t)\end{equation}$$

Note that objective $$\ref{orig_obj}$$ and $$\ref{inter1}$$ are different and usually have different values, but the optimization on the two are equivalent in terms of improving expected rewards.

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
