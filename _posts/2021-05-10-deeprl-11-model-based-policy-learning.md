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

On the other hand, if we have an explicit policy, we can make decision at each time step based on the state at that time step, and therefore no need to plan the whole action sequence all in one go. This is closed-loop planning and it's more desirable in the stochastic dynamics setting.

## 1 Learn Policy by Backprop Through Time (BPTT)
Suppose we have learned dynamics $$s_{t+1} = f(s_t, a_t)$$ and reward function $$r(s_t, a_t)$$, and want to learn optimal policy $$a_t = \pi_{\theta}(s_t)$$. (Here is use deterministic dynamics and policy to make a point. The point also applies to stochastic dynamics, but the derivation is slightly more involved and will be introduced in the future; Also I drop the parameters notation in the dynamics and reward function for simplicity). Same as policy gradient, our goal will be:

$$\begin{align}
\theta^* = \text{argmax}_{\theta} \mathbb{E}_{\tau\sim p(\tau)}\sum_t r(s_t, a_t)
\end{align}$$

Since we have dynamics and reward function, we can write the objective as 

$$\begin{align}
\mathbb{E}_{\tau\sim p(\tau)}\sum_t r(s_t, a_t) = \sum_t r(f(s_{t-1}, a_{t-1}), \pi_{\theta}(f(s_{t-1}, a_{t-1}))), \text{ where } s_{t-1} = f(s_{t-2}, a_{t-2})
\end{align}$$

Very similar to shooting methods, the objective is defined recursively, which lead to high sensitivity to the first actions and lead to poor numerical stability. However, for shooting methods, if we define the process as LQR, we can use a dynamical programming to solve it in a very stable fashion. Unfortunately, unlike LQR, since the the parameters of the policy couple all time steps, we cannot solve by dynamical programming (i.e. can't calculate the best policy parameter for the last time step and solve for the policy parameters for second to last time step and so on).

What we can use is backpropagation:
<div align="center"><img src="../assets/images/285-11-bptt.png" width="700"></div>

If you are familiar with recurrent neural network, we might realize that the kind of backprop shown above is the so called Backpropagation Through Time or BPTT, which is usually used on recurrent neural nets like LSTM. BPTT famously has the vanishing or exploding gradients issue, because all the jacobians of different time steps get multiplied together. This issue can only get worse in policy learning, because in sequence deep learning, we can choose architectures like LSTM that has good gradient behavior, but in model-based RL, the dynamics has to fit to the data and we don't have control over the gradient behavior.

In the next two sections, we will introduce two popular ways to model-based RL. The first way is a bit controvesal, it does model-free optimization (policy gradient, actor-critic, Q-learning etc.) and use model to only generate sythetic data. Despite looking weird backwards, this idea can work very well in pratice; the second way is to **________________**

## 2 Model-Free Optimization with a Model
Reinforcement learning is about getting better by interacting with the world, and the interacting, try-and-error process can be time consuming (even in a simulator sometimes). If we have a mathematical model that represent how the world works, then we can effortlessly generate data (transitions) from it for model-free algorithms to get better. However, it's impossible to have a comprehansive mathematical model of the world, or even of the environment we want to run our RL algorithms. Nevertheless, a learned dynamics is a representation of the environment and we can use it to generate data.

The general idea is to use the learned dynamics to provide more training data for model-free algorithms, it does it by generate model-based rollouts from real world states. 

The general algorithm is the following:

1. run *some* policy in the environment to collect data $$\mathcal{B}$$
2. sample minibatch $${(s, a, r, s')}$$ from $$\mathcal{B}$$ uniformly
3. use $${(s, a, r, s')}$$ to update model and reward function
4. sample minibatch {(s)} from $$\mathcal{B}$$
5. for each $$s$$, perform model-based **k** step rollout with neural net policy $$\pi_{\theta}$$ or policy indiced from Q-learning , get transitions $${(s, a, r, s')}$$
6. use the transitions for policy gradient update or updating Q-function. Go to step 4 for a few more times (inner loop); Go to step 1 (outer loop).

There are a few things to be cleared. Above algorithm is very general and explicitly considers both policy gradient and Q-learning, this will affect what we actually do in step 1, 5, and 6. If we use policy gradient, in step 1 and step 5, we can run the learned policy, and in step 6, we run policy gradient update; if we use Q-learning, then in step 1 and step 5, we run policy indiced by the learned Q-function, e.g. $$\epsilon$$-greedy policy. And in step 6 we update the Q-function by taking the gradient of temporal difference. 

Model-based rollout step **k** is an very important hyperparameter, since we completely reply on $$f_{\phi}(s_t, a_t)$$ during model-based rollout, the discrepancy between $$f_{\phi}(s_t, a_t)$$ and the ground truth dynamics can result in a distribution shift problem, i.e. the expectation in the objective we are optimizing is over a distribution that is very different from the true distribution. We've encounter this issue several times (e.g. imitation learning, TRPO etc.). We know that if there is a discrepancy between fitted dynamics and true dynamics, the error between true objective and the objective we optimizes grow linearly with the length of rollout. Therefore, we don't want the length of model-based rollout to be too long; on the other hand, too short a rollout provide little learning signal, which is undesirable for policy update or Q-function update. Therefore, we need to choose an appropriate k for the algorithm.

Since for every model-based rollout, the initial state is sampled from real world data, this algorithm can be intuitively understand as imagining different possibilities starting from real world situations:

<div align="center"><img src="../assets/images/285-11-branch.png" width="500"></div>


Here we give one instatiation of the general algorithm introduced above, which combines model-based RL with policy gradient methods. The algorithm is called Model-Based Policy Optimization or MBPO [Janner et al. 19'](https://arxiv.org/pdf/1906.08253.pdf):

<div align="center"><img src="../assets/images/285-11-mbpo.png" width="700"></div>