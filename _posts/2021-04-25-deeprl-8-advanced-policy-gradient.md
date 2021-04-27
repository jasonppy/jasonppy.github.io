---
title: "Deep RL 8 Advanced Policy Gradient"
date: 2021-04-25
categories:
  - DeepRL
tags:
  - RL
  - Notes
---
At the end of previous lecture, we talked about the issues with Q-learning, one of them is that it's not directly optimizing the expected return and it can take a long time before the return starts to improve. On the other hand, policy gradient methods are direclty optimizing the expected return, although we cannot guarantee that the return will improve every gardient update. At the same time, we know that classic policy iteration can improve the expected return at each iteration, but this method cannot be applied to large scale problems.

In this section, we derive stable policy gradient methods, by firstly framing them as policy iteration.

## 1 Policy Gradient as Policy Iteration
Let's write down the difference between expected return under previous policy $$\pi_{\theta}$$ and under new (updated) policy $$\pi_{\theta'}$$: 

$$\begin{align}
&J(\theta') - J(\theta)\\
&= \mathbb{E}_{\tau \sim p_{\theta'}(\tau)}\left[ \sum_{t=0}^{\infty} \gamma^tr(s_t, a_t) \right] - \mathbb{E}_{\tau \sim p_{\theta}(\tau)}\left[ \sum_{t=0}^{\infty} \gamma^tr(s_t, a_t) \right] \\
&= \mathbb{E}_{\tau \sim p_{\theta'}(\tau)}\left[ \sum_{t=0}^{\infty} \gamma^tr(s_t, a_t) \right] - \mathbb{E}_{s_0 \sim p(s_0)}\left[ V^{\pi_{\theta}}(s_0) \right] \\
&= \mathbb{E}_{\tau \sim p_{\theta'}(\tau)}\left[ \sum_{t=0}^{\infty} \gamma^tr(s_t, a_t) \right] - \mathbb{E}_{\tau \sim p_{\theta'}(\tau)}\left[ V^{\pi_{\theta}}(s_0) \right] \\
&= \mathbb{E}_{\tau \sim p_{\theta'}(\tau)}\left[ \sum_{t=0}^{\infty} \gamma^tr(s_t, a_t) \right] - \mathbb{E}_{\tau \sim p_{\theta'}(\tau)}\left[ V^{\pi_{\theta}}(s_0) + \sum_{t=1}^{\infty}\gamma^{t}V^{\pi_{\theta}}(s_{t}) - \sum_{t=1}^{\infty}\gamma^{t}V^{\pi_{\theta}}(s_{t}) \right] \\
&= \mathbb{E}_{\tau \sim p_{\theta'}(\tau)}\left[ \sum_{t=0}^{\infty} \gamma^tr(s_t, a_t) \right] + \mathbb{E}_{\tau \sim p_{\theta'}(\tau)}\left[ \sum_{t=0}^{\infty}\gamma^{t}(\gamma V(s_{t+1}) - V^{\pi_{\theta}}(s_{t})) \right] \\
&= \mathbb{E}_{\tau \sim p_{\theta'}(\tau)}\left[ \sum_{t=0}^{\infty} \gamma^t(r(s_t, a_t)\gamma V(s_{t+1}) - V^{\pi_{\theta}}(s_{t})) \right] \\
&= \mathbb{E}_{\tau \sim p_{\theta'}(\tau)}\left[ \sum_{t=0}^{\infty} \gamma^t A^{\pi_{\theta}}(s_t, a_t) \right] 
\end{align}$$

Here we have proved an intersting equality:

$$\begin{equation}\label{diff}
J(\theta') - J(\theta) = \mathbb{E}_{\tau \sim p_{\theta'}(\tau)}\left[ \sum_{t=0}^{\infty} \gamma^t A^{\pi_{\theta}}(s_t, a_t) \right] 
\end{equation}$$

The difference of expected return equals to the expected value of the advantage of the previous policy $$\pi_{\theta}$$ under the trajectory distribution of the new policy $$\pi_{\theta'}$$. 

Note that we haven't done any policy gradient specific operation, so this equality is universal. We can use this to understand why policy iteration improve the expected return at every iteration i.e. $$J(\theta') - J(\theta) \geq 0$$: in policy iteration, the policy is deterministic and updated as $$\pi'(s) = \text{argmax}_a A^{\pi}(s_t, a_t)$$. Therefore when the $$s_t, a_t$$ are from the new policy $$\pi'$$, we always have $$A^{\pi_{\theta}}(s_t, a_t) \geq 0$$, and thus $$J(\theta') - J(\theta) \geq 0$$.

Now let's consider how to have this monotonic improvement in expected return in policy gradient methods. Well, this cannot be guaranteed theoretically because we need to introduce some approximation in order to derive a policy gradient algorithm from equation $$\ref{diff}$$. Nevertheless, the resulting method --- TRPO --- is the first stable RL algorithm in that during training the return will improve gradually (whereas another popular methods at the time --- DQN --- is very unstable).

## 2 Trust Region Policy Optimization (TRPO)
As a policy gradient method, TRPO aims at directly maximizing equation $$\ref{diff}$$, but this cannot be done because the trajectory distribution is under the new policy $$\pi_{\theta'}$$. The sample trajectories that we have come from the previous policy $$\pi_{\theta}$$. Therefore we introduce the approximation:



## Demo: TRPO
<iframe width="1424" height="652" src="https://www.youtube.com/embed/KJ15iGGJFvQ" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>