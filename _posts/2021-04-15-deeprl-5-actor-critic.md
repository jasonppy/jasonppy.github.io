---
title: "Deep RL 5 Actor Critic"
date: 2021-04-15
categories:
  - DeepRL
tags:
  - RL
  - Notes
---
Actor-critic algorithms build on the policy gradient framwork that we discussed in the previous lecture, but also augment it with learning value functions and Q-functions. The goal of this augmentation is still reducing variance, but from a slightly different angle.

Let's take a look at the original policy gradient and it's Monte Carlo approximator:

$$\begin{align}
\nabla_{\theta}J(\theta) &= \mathbb{E}_{\tau\sim p_{\theta}(\tau)}\nabla_{\theta}\log \pi_{\theta}(a_t\mid s_t)r(s_t, a_t) \label{orig} \\
&\approx \frac1N \sum_{i=1}^{N}\left(\sum_{t=1}^{T}\nabla_{\theta}\log \pi_{\theta}(a^i_t\mid s^i_t)\right) \left(\sum_{t=1}^T r^i_t\right) \label{approx} \\
\end{align}$$

In equation $$\ref{approx}$$, $$N$$ sample trajectories is used to approximate the expectation $$\mathbb{E}_{\tau\sim p_{\theta}(\tau)}$$, but in equation $$\ref{approx}$$ there are still one quantity that are approximated, i.e. the reward function $$r(r^i_t, a^i_t)$$, and unfortuately, we are only using one sample $$r^i_t$$ to approximate it. 

Even with the use of causality and baseline, which gives

$$\begin{align}\label{var_red}
\nabla_{\theta}J{\theta}&\approx\frac1N \sum_{i=1}^{N}\sum_{t=1}^{T}\nabla_{\theta}\log \pi_{\theta}(a^i_t\mid s^i_t) \left(\sum_{t'=t}^T r^i_t - b\right)
\end{align}$$

with $$b = \frac1N \sum_{i=1}^N\sum_{t'=t}^{T}r^i_t$$, we are improving the variance in terms of expectation over trajectories ($$\mathbb{E}_{\tau\sim p_{\theta}(\tau)}$$) but not the estimation of *reward to go* (or better-than-average reward to go).

Actor-critic algorithms aims at better estimating the reward to go.

## 1 Fit the Value Function
We start by recalling the goal of reinforcement learning:

$$\begin{equation}\label{goal2}\text{argmax} \mathbb{E}_{p(\tau)}\sum_{t=1}^T r(s_t, a_t)\end{equation}$$

Now define the *Q-function*:

$$\begin{align} Q^{\pi}(s_t,a_t) &= \mathbb{E}_{p_{\theta}}\left[\sum_{t'=t}^{T}r(s_{t'},a_{t'}) \mid s_t, a_t \right] \\
&= r(s_t, a_t) + \mathbb{E}_{a_{t+1} \sim \pi_{\theta}(a_{t+1}\mid s_{t+1}),s_{t+1}\sim p(s_{t+1}\mid s_t, a_t)} \left[ Q^{\pi}(s_{t+1}, a_{t+1}) \right] 
\end{align}$$

Q-function is exactly the expected reward to go from step $$t$$ given the state and action $$(s_t, a_t)$$. 

How about the baseline $$b$$ in $$\ref{var_red}$$? We can also replace it with lower variance estimate. To do that, we define the *value function*:

$$\begin{align}
V^{\pi}(s_t) & = \mathbb{E}_{p_{\theta}}\left[\sum_{t'=t}^{T}r(s_{t'},a_{t'}) \mid s_t \right] \\
&= \mathbb{E}_{a_t\sim \pi_{\theta}(a_t\mid s_t)} \left\{ r(s_t, a_t) + \mathbb{E}_{s_{t+1} \sim p(s_{t+1}\mid s_t, a_t)}\left[ V^{\pi}(s_{t+1}) \right] \right\} \end{align}$$

Value function measures how good the state is (i.e. the *value* of the state). This is exactly the expected reward of state averaged over different actions.

In addition, we define the *advantage*

$$\begin{equation}
A^{\pi}(s_t, a_t) = Q^{\pi}(s_t, a_t) - V^{\pi}(s_t)
\end{equation}$$

In fact, $$\sum_{t'=t}^T r^i_t - \frac1N \sum_{i=1}^N\sum_{t'=t}^{T}r^i_t$$ in equation $$\ref{var_red}$$ looks very much just like the one sample estimate of the advantage $$A^{\pi}(s_t, a_t)$$. In fact, $$\sum_{t'=t}^T r^i_t$$ is an one sample estimate of The Q-function $$Q^{\pi}(s_t, a_t)$$, however, $$\frac1N \sum_{i=1}^N\sum_{t'=t}^{T}r^i_t$$ is not a estimate of $$V^{\pi}(s_t)$$ (actually if the state space is continuous, we will only have an one sample estimate of $$V^{\pi}(s_t)$$, which is also the $$\sum_{t'=t}^T r^i_t$$, which is the same as the one sample estimate of $$Q^{\pi}(s_t, a_t)$$. We only have an one sample estimate because we will never visit the state again). But $$V^{\pi}(s_t)$$ is intuitively better than $$\frac1N \sum_{i=1}^N\sum_{t'=t}^{T}r^i_t$$ even if we don't consider the variance, because the former is the expected reward for state $$s_t$$, and the later is an estimate of expected reward at time step $$t$$ averaged over all possibly state. Since we want to know how good the action is in the current *state*, rather than in the current *time step*, we prefer the former.













If we can use many sample trajectories and the corresponding rewards to train a neural network to fit the Q-function, then estimated Q-function could be a lower variance alternative than the one sample estimate $$\sum_{t=1}^T r^i_t$$ in equation $$\ref{var_red}$$.

Same as the Q-function, we can use a bunch of sample trajectories and rewards to train a neural network to estimate it.

Suppose we have trained two neural nets to approximate Q-function and value function separately, we can then replace them with the one sample estimates in equation $$\ref{var_red}$$ and therefore get a 




First notice that Q-function and value function are related by

$$\begin{equation}\label{relation}V(s_t) = \mathbb{E}_{a_t\sim \pi(a_t\mid s_t)}Q(s_t, a_t)\end{equation}$$

Equation $$\ref{relation}$$ gives the a nice intuition about Q-function and value function --- value function evaluates on average how different actions  at the the current state is. This leads to another idea improve the explicit policy --- we improve policy $$\pi_{\theta}$$ such that the actions taken by running the policy is better than average, i.e. the probability that $$Q(s_t, a_t) > V(s_t)$$ is high. This leads to the actor-critic algorithm which stands at the intersection of policy gradient methods and value-based methods. Note that we can also derive Q-function from value function:

$$\begin{equation} Q(s_t, a_t) = r(s_t, a_t) + \mathbb{E}_{s_{t+1} \sim p(s_{t+1}\mid s_t, a_t)}V(s_{t+1})\end{equation}$$

