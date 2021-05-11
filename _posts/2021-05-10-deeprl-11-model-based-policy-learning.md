---
title: "Deep RL 11 Model-Based Policy Learning"
date: 2021-05-10
categories:
  - DeepRL
tags:
  - RL
  - Notes
---
In this section, we study how to learn policies utilize the known (learned) dynamics. Why do we need to learn a policy? What's wrong with MPC in the previous lecture? The answer is that MPC is still an open loop control methods, even though the replanning machanism provides some amount of closed-loop capability, but the planning procedure still is unable to reason under the fact that more information will be revealed in the future and we can act based on that information. This is obviously suboptimal in the stochastic dynamics setting.

On the other hand, if we have an explicity policy, we can make decision at each time step based on the state at that time step, and therefore no need to plan the whole action sequence all in one go. This is closed-loop planning and it's more desirable in the stochastic dynamics setting.

Suppose we have learned dynamics $$s_{t+1} = f(s_t, a_t)$$ and reward function $$r(s_t, a_t)$$ (Here I just want to make a point, so for simplicity I use deterministic dynamics. The point also applies to stochastic dynamics, but the derivation is slightly more involved and will be introduced in the future; 
Also I drop the parameters notation in the dynamics and reward function for simplicity), and want to learn optimal policy $$a_t = \pi_{\theta}(s_t)$$ (for the same reason I use a deterministic policy). Same as policy gradient, our goal will be:

$$\begin{align}
\theta^* = \text{argmax}_{\theta} \mathbb{E}_{\tau\sim p(\tau)}\sum_t r(s_t, a_t)
\end{align}$$

But onlike policy gradient, since we have dynamics and reward function, we can write the objective as 

$$\begin{align}
\mathbb{E}_{\tau\sim p(\tau)}\sum_t r(s_t, a_t) = \sum_t r(f(s_{t-1}, a_{t-1}), \pi_{\theta}(f(s_{t-1}, a_{t-1}))), \text{ where } s_{t-1} = f(s_{t-2}, a_{t-2})
\end{align}$$

Very similar to shooting methods, the objective is defined recursively, which lead to high sensitivity to the first actions and lead to poor numerical stability. However, for shooting methods, if we define the process as LQR, we can use a dynamical programming to solve it in a very stable fashion. Unfortunately, unlike LQR, since the the parameters of the policy couple all time steps, we cannot solve by dynamical programming (i.e. can't calculate the best policy parameter for the last time step and solve for the policy parameters for second to last time step and so on).

What we can use is backpropagation:
<div align="center"><img src="../assets/images/285-11-bptt.png" width="700"></div>

If you are familiar with Deep Learning, we might realize that this is backpropagation through time, which is usually used on recurrent neural nets like LSTM. BPTT famously has the vanishing or exploding gradients issue because all the jacobians of different time steps get multiplied together. This issue can only get worse in policy learning, because in sequence deep learning, we can choose architectures like LSTM that has good gradient behavior while in model-based RL, the dynamics has to fit to the data and we don't have control over the gradient behavior.