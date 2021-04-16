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

If we have Q-function and value function, we can plug them in the original policy gradient and get the most ideal estimator:

$$
\nabla_{\theta}J(\theta) = \frac1N\sum_{i=1}^N\sum_{t=1}^T \nabla_{\theta}\pi_{\theta}(a^i_t\mid s^i_t)\left(Q^{\pi}(s^i_t, a^i_t) - V^{\pi}(s^i_t)\right)
$$

However, we do not have $$Q^{\pi}(s_t, a_t)$$ or $$V^{\pi}(s_t)$$ and therefore we want to estimate them. Instead of using Monte Carlo estimate, we use function approximation, which might lead to a biased estimatio, but will give enormous variance reduction, in practice, the later usually brings more benefits than the hurts the former brings.

So we want to fit two neural networks to approximate $$Q^{\pi}(s_t, a_t)$$ or $$V^{\pi}(s_t) separately$$? Well, it's actually not necessary if we notice the relationship between the two functions:

$$\begin{equation}\label{relation} Q^{\pi}(s_t, a_t) = r(s_t, a_t) + \mathbb{E}_{s_{t+1} \sim p(s_{t+1}\mid s_t, a_t)}V^{\pi}(s_{t+1})\end{equation}$$

And in practice we use one sample estimation to approximate $$\mathbb{E}_{s_{t+1} \sim p(s_{t+1}\mid s_t, a_t)}$$, then we have 

$$\begin{equation} Q^{\pi}(s^i_t, a^i_t) \approx r^i_t + V^{\pi}(s^i_{t+1})\end{equation}$$

Therefore, we only need to fit $$V^{\pi}$$. For training data, since $$V^{\pi}(s_t)=\mathbb{E}_{p_{\theta}}\left[\sum_{t'=t}^{T}r(s_{t'},a_{t'}) \mid s_t \right]$$, ideally for every state $$s_t$$, we want to have a bunch of differernt rollouts and rewards collected starting from that state, and then use the sample mean of rewards as the estimate of $$V^{\pi}(s_t)$$. But this require reseting the simulator and actually impossible in the real world. Therefore, we have one sample estimator $$\sum_{t'=t}^Tr_{t'}$$,

So, we use training data $$\{ (s^i_t, \sum_{t'}^Tr^i_{t'}) \}_{i=1,t=1}^{N, T}$$
to train a neural network $$V^{\pi}_{\phi}$$ in a supervised way. But we can further reduce the variance even if we only have one sample estimation of the expected reward --- we can again apply the function approximation idea and replace $$\sum_{t'}^Tr^i_{t'}$$ with $$r^i_t + V^{\pi}_{\phi'}(s^i_{t+1})$$ in the training data, where $$V^{\pi}_{\phi'}$$ is the previously fitted value function (i.e. $$\phi'$$ is one gradient step before $$\phi$$). $$r^i_t + V^{\pi}_{\phi'}(s^i_{t+1})$$ is called the *Bootstrapp estimate* of the value function. In summary, we fit $$V^{\pi}_{\phi}$$ to $$V^{\pi}$$ by minimizing

$$\begin{equation}\label{value_obj}
\frac{1}{NT}\sum_{i,t=1,1}^{N,T}\left\|V^{\pi}_{\phi}(s^i_t) - y^i_t\right\|^2
\end{equation}$$

where $$y^i_t = \sum_{t'}^Tr^i_{t'}$$ or $$r^i_t + V^{\pi}_{\phi'}(s^i_{t+1})$$ and the later usually works better.


With fitted value function $$V^{\pi}_{\phi}$$, we can estimate the Q-function (Q-value) by 

$$Q^{\pi}_{\phi}(s^i_t, a^i_t) \approx r^i_t + V^{\pi}_{\phi}(s^i_{t+1})$$

and therefore the advantage:

$$A^{\pi}_{\phi}(s^i_t, a^i_t) = r^i_t + V^{\pi}_{\phi}(s^i_{t+1}) - V^{\pi}_{\phi}(s^i_t)$$

And our actor-critic policy gradient is

$$\begin{align}
\nabla_{\theta}J(\theta) 
&= \frac1N\sum_{i=1}^N\sum_{t=1}^T \nabla_{\theta}\pi_{\theta}(a^i_t\mid s^i_t)\left(Q^{\pi}(s^i_t, a^i_t) - V^{\pi}(s^i_t)\right) \\
&= \frac1N\sum_{i=1}^N\sum_{t=1}^T \nabla_{\theta}\pi_{\theta}(a^i_t\mid s^i_t)\left(r(s^i_t, a^i_t) + \mathbb{E}_{s_{t+1} \sim p(s_{t+1}\mid s^i_t, a^i_t)}V^{\pi}(s_{t+1}) - V^{\pi}(s^i_t)\right) \\
&\approx \frac1N\sum_{i=1}^N\sum_{t=1}^T \nabla_{\theta}\pi_{\theta}(a^i_t\mid s^i_t)\left(r^i_t + V^{\pi}_{\phi}(s^i_{t+1}) - V^{\pi}_{\phi}(s^i_t)\right) \\
&=\frac1N\sum_{i=1}^N\sum_{t=1}^T \nabla_{\theta}\pi_{\theta}(a^i_t\mid s^i_t)A^{\pi}_{\phi}(s^i_t, a^i_t)
\end{align}$$


The *batch* actor-critic algorithm is:

1. run current policy $$\pi_{\theta}$$ and get trajectories $$\{ \tau^i \}_{i=1}^{N}$$ and rewards $$\{ r^i_t \}_{i,t=1,1}^{N,T}$$
2. fit value function $$V^{\pi}_{\phi}$$ by minimizing equation $$\ref{value_obj}$$ 
3. calculate the advantage of each state action pair $$A^{\pi}_{\phi}(s^i_t, a^i_t) = r^i_t + V^{\pi}_{\phi}(s^i_{t+1}) - V^{\pi}_{\phi}(s^i_t)$$
4. calculate actor-critic policy gradient $$\nabla_{\theta}J(\theta) =\frac1N\sum_{i=1}^N\sum_{t=1}^T \nabla_{\theta}\pi_{\theta}(a^i_t\mid s^i_t)A^{\pi}_{\phi}(s^i_t, a^i_t)$$
5. gradient update: $$\theta \leftarrow \theta + \alpha\nabla_{\theta}J(\theta)$$

We call it *batch* in that for each policy update we collect a batch of trajectories. We can also update the policy (and value function) using only one step of data i.e. $$(s_t, a_t, r_t, s_{s+1})$$, which leads to the *online* actor-critic algorithm which we will introduce later.

## 2 Discount Factor
Our previous discussion on policy gradient and actor-critic algorithms are all within the finite horizon or episodic learning scenario, where there is an ending time step $$T$$ for the task. What about the infinite horizon scenario i.e. $$T = \infty$$?

Well in that case the original algorithm can run into problems because at the second step, $$V^{\pi}_{\phi}$$ can get infinitely large in many cases. Or in vanilla policy gradient method, the sum of reward can get infinitely large. 

To remedy that, we introduce the discount factor $$\gamma \in (0,1)$$ and define the discounted expected reward, value function, and Q-function to be

$$\begin{align}
J(\theta) &= \mathbb{E}_{\tau\sim p_{\theta}(\tau)}\sum_t \gamma^t r(s_t, a_t)\\
V^{\pi}(s_t) &= \mathbb{E}_{\tau_t \sim p_{\theta}(\tau_t)}\left[ \sum_{t'=t} \gamma^{t'-t}r(s_t', a_t') \mid s_t \right]\\
Q^{\pi}(s_t, a_t) &= \mathbb{E}_{\tau_t \sim p_{\theta}(\tau_t)}\left[ \sum_{t'=t} \gamma^{t'-t}r(s_t', a_t') \mid s_t, a_t \right]\\
&= r(s_t, a_t) + \gamma\mathbb{E}_{s_{t+1} \sim p(s_{t+1}\mid s_t, a_t)}V^{\pi}(s_{t+1})
\end{align}$$

Therefore, the policy gradient and actor-critic policy gradient are
$$\begin{align}
\nabla_{\theta}J_{\text{PG}}(\theta) &= \frac1N \sum_{i=1}^N\sum_{t=1}\nabla_{\theta}\log \pi_{\theta}(a^i_t\mid s^i_t)\left(\sum_{t'=t}\gamma^{t'-t}r^i_{t'} - b\right) \label{dis_pg}\\
\nabla_{\theta}J_{\text{AC}}(\theta) &= \frac1N \sum_{i=1}^N\sum_{t=1}\nabla_{\theta}\log \pi_{\theta}(a^i_t\mid s^i_t)\left(r^i_t + \gamma V^{\pi}_{\phi}(s^i_{t+1}) - v^{\pi}_{\phi}(s^i_t)\right) \label{dis_ac}
\end{align}$$

where in $$J_{\text{PG}}(\theta)$$, $$b$$ is also a discounted baseline e.g. $$\frac1N\sum_{i=1}^N\sum_{t'=t}\gamma^{t'-t}r^i_{t'}$$.

Usually we set $$\gamma$$ to be something like $$0.99$$. It can be proved that discount factor can prevent the expected reward from being infinity and reduce the variance. Therefore, actually in most cases, no matter it's finite of infinite horizon, people use discounted policy gradient, i.e. equation $$\ref{dis_pg} \text{ and } \ref{dis_ac}$$, rather than the original ones.

## 3 Online Actor-critic algorithms


## 4 More tricks
### Critic as baseline
why state dependent baseline still gives unbiased gradient estimator?

$$\begin{align}
\nabla_{\theta}J(\theta) = \mathbb{E}_{\tau\sim p_{\theta}(\tau)}\sum_{t=1}^T\nabla_{\theta}\log \pi_{\theta}(a_t\mid s_t)\left( \sum_{t'=t}^{T}\gamma^{t'-t}r(s_{t'}, a_{t'}) - V^{\pi}(s_t)\right)
\end{align}$$

let's take one element of the summation over time horizon out:

$$\begin{align}
\nabla_{\theta}J(\theta)_t &= \mathbb{E}_{\tau_t\sim p_{\theta}(\tau_t)}\nabla_{\theta}\log \pi_{\theta}(a_t\mid s_t)\left( \sum_{t'=t}^{T}\gamma^{t'-t}r(s_{t'}, a_{t'}) - V^{\pi}(s_t)\right) \\
&= \mathbb{E}_{\tau_t\sim p_{\theta}(\tau_t)}\nabla_{\theta}\log \pi_{\theta}(a_t\mid s_t)\left( \sum_{t'=t}^{T}\gamma^{t'-t}r(s_{t'}, a_{t'}) \right)- \mathbb{E}_{\tau_t\sim p_{\theta}(\tau_t)}\nabla_{\theta}\log \pi_{\theta}(a_t\mid s_t)V^{\pi}(s_t) \\
&=  \mathbb{E}_{\tau_t\sim p_{\theta}(\tau_t)}\nabla_{\theta}\log \pi_{\theta}(a_t\mid s_t)\left( \sum_{t'=t}^{T}\gamma^{t'-t}r(s_{t'}, a_{t'}) \right) - \mathbb{E}_{s_{1:t}, a_{1:t-1}}V^{\pi}(s_t) \mathbb{E}_{a_t \sim \pi_{\theta}(a_t\mid s_t)}\nabla_{\theta}\log \pi_{\theta}(a_t\mid s_t) \\
&=  \mathbb{E}_{\tau_t\sim p_{\theta}(\tau_t)}\nabla_{\theta}\log \pi_{\theta}(a_t\mid s_t)\left( \sum_{t'=t}^{T}\gamma^{t'-t}r(s_{t'}, a_{t'}) \right) - \mathbb{E}_{s_{1:t}, a_{1:t-1}}V^{\pi}(s_t) \cdot 0 \\
&=  \mathbb{E}_{\tau_t\sim p_{\theta}(\tau_t)}\nabla_{\theta}\log \pi_{\theta}(a_t\mid s_t)\left( \sum_{t'=t}^{T}\gamma^{t'-t}r(s_{t'}, a_{t'}) \right)
\end{align}$$