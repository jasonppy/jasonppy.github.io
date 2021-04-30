---
title: "Deep RL 9 Model-based Planning"
date: 2021-04-29
categories:
  - DeepRL
tags:
  - RL
  - Notes
---

Let's recall the reinforcement learning goal --- we want to maximaze the expected reward (or discounted reward in the infinite horizon case)

$$\begin{equation}
\mathbb{E}_{\tau\sim p(\tau)}\sum_{t=1}^T r(s_t, a_t)
\end{equation}$$

where 

$$\begin{equation}
p(\tau) = p(s_1)\prod_{t=1}^{T}p(s_{t+1}\mid s_t, a_t)\pi(a_t\mid s_t)
\end{equation}$$

In most methods that we've introduced so far, such as policy gradient, actor-critic, Q-learning, etc. the transition dynamics $$p(s_{t+1}\mid s_t, a_t)$$ is assumed to be unknown. But in many cases, the dynamics is actually known to us, such as the game of Go (we know what the board will look like after we make a move), Atari games, car navigation, anything in simulated environments (although we may not want to utilize the dynamics in this case) etc.

Knowing the dynamics provides addition information, which in principle should improve the actions we take. In this lecture, we study how to plan actions to maximize the expected reward when the dynamics is known. We will mostly study deterministic dynamics, i.e. $$s_{t+1} = f(s_t, a_t)$$. Although we will also generalize some methods to stochastic dynamics, i.e. $$s_{t+1} \sim p(s_{t+1}\mid s_t, a_t)$$.

## 1 Open-loop planning
If we know the deterministic dynamics, then giving the first state $$s_1$$, we should be able to know all the remaining states given the actions sequence (and therefore the rewards). Open-loop planning aims at directly giving optimal actions sequences without waiting for the trajectory to unfold.
<div align="center"><img src="../assets/images/285-10-open.png" width="700"></div>

**Random Shooting Method**

**Cross-entropy Method**

However, even though extremely simple and can be parallelized easily, this two methods have very harsh dimensionality limit, and open-loop plan itself can be suboptimal in stochastic case.
