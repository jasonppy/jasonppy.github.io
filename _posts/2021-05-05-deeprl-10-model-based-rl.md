---
title: "Deep RL 10 Model-based Reinforcement Learning"
date: 2021-05-05
categories:
  - DeepRL
tags:
  - RL
  - Notes
---
Previous lecture is mainly about how to plan actions to take when the dynamics is known. In this lecture, we study how to learn the dynamics. We will also introduce how to incorporate planning in the model learning process and therefore form a complete decision making algorithm.

Again, most of the algorithms will be introduced in the context of deterministic dynamics, i.e. $$s_{t+1} = f(s_t, a_t)$$, but almost all of these algorithms can just as well be applied in the stochastic dynamics setting, i.e. $$s_{t+1}\sim p(s_{t+1}\mid s_t, a_t)$$, and when the distinction is salient, I'll make it explicit.

## 1 Basic Model-based RL 
How to learn a model, the most direct way is supervised learning. Similar to the idea used before, we run a random policy to get transitions, and then fit a neural net to the transition:

1. run base policy $$\pi_0(a\mid s)$$ (e.g. random policy) to collect $$\mathcal{D} = \{ (s_i, a_i, s'_i) \}$$
2. learn dynamics model $$f_{\theta}(s,a)$$ to minimize $$\sum_{i}\left\| s'_i - f_{\theta}(s_i, a_i) \right\|^2$$
3. plan through $$f_{\theta}(s,a)$$ to choose actions.

Where in step 3, we can use CEM, MCTS, LQR etc. 

Does this work? Well, in some cases. For example, if we have a full physics model of the dynamics and only need to fit a few parameters, this method can work. But still, some care should be taken to design a good base policy.

In general, however, this doesn't workm and the reason is very similar to the one we encountered in imitation learning --- distribution shift. 
The data we used to learn the dynamics comes from the trajectory distribution induced by random policy $$\pi_0$$, but when we plan through the model, we can think of the algorithm is using another policy $$\pi_f$$, and the trajectory distribution induced by this policy can be very different from the one induced by the base policy. The consequence is that, when we plan actions, we'll arrive at the state action pair that the dynamics is very uncertain about, because it has never trained on similar data! Therefore, it will make bad prediction on the following state, which will in turn lead to bad actions, this will go on and on and in the end we are completely planning on the wrong states (prediction is different the reality). The intuitive plot is shown below:

<div align="center"><img src="../assets/images/285-10-mismatch.png" width="700"></div>

How to deal with it? Same as how DAgger deals with distribution shift in imitation learning, we just need to make sure that the training data comes from the current  dynamics (current policy). These lead to the first practical model-based RL algorithm:

1. run base policy $$\pi_0(a\mid s)$$ (e.g. random policy) to collect $$\mathcal{D} = \{ (s_i, a_i, s'_i) \}$$
2. learn dynamics model $$f_{\theta}(s,a)$$ to minimize $$\sum_{i}\left\| s'_i - f_{\theta}(s_i, a_i) \right\|^2$$
3. plan through $$f_{\theta}(s,a)$$ to choose actions.
4. execute those actions and add the resulting transitions $$\{ (s_j, a_j, s'_j) \}$$ to $$\mathcal{D}$$. Go to step 2.

However, even though the data is updating based on the learned dynamics, as long as we are replanning, it will always induce a new trajectory distribution which will be a little different from the previous distribution. In another word, the distribution shift will always exist. Therefore, as we plan through $$f_{\theta}(s,a)$$, the actual trajectory will gradually deviate from the predicted trajectory which will lead to bad actions. 

We can improve this algorithm by only execute the first planned action, and observe the next state that this action leads to, and then replan start from that state. and then take the first action etc. In a word, at each step, we only take the first planned action and observe the state and then replan from there. Because at each time step, we always take the action based on the actual state, this is more reliable than executing the whole plan actions all in one go. The algorithm is

1. run base policy $$\pi_0(a\mid s)$$ (e.g. random policy) to collect $$\mathcal{D} = \{ (s_i, a_i, s'_i) \}$$
2. learn dynamics model $$f_{\theta}(s,a)$$ to minimize $$\sum_{i}\left\| s'_i - f_{\theta}(s_i, a_i) \right\|^2$$
3. plan through $$f_{\theta}(s,a)$$ to choose actions.
4. execute the first planned action and add the resulting transition $$(s, a, s)$$ to $$\mathcal{D}$$. If reach the predefined maximal number of planning steps, go to step 2; else, Go to step 3. 

This algorithm is call Model Predictive Control or MPC. Replanning at each time step can drastically increase the computation load, so people sometimes choose to shorten the time horizon of the trajectory. While this might lead to a decrease in the quality of actions, since we are constantly replanning, we can take the cost that individual plans is less perfect.

## Uncertainty-Aware Model-based RL
Since we plan actions replying on the fitted dynamics, whether or not the dynamics is a good representation of the world is crucial. When we use high capacity model like neural networks, we usually need to feed it with a lot of data in order to get a good fit. But in model-based RL, we usually don't have a lot of data at the beginning, in fact, we can only have some bad data (generated by running some random policy), and then if we use neural network to fit the dynamics, it will overfit the data, and not have a good representation of the good part of the world. This will lead the algorithm to take bad actions, which can lead to bad states, which can then lead to neural net dynamics trained only on trajectories and thus it's predictions on good states in unreliable, which again lead to algorithm to take bad actions...... This seems to be a chicken-and-egg problem, but if you think about it, the origin is that planning on unconfident state prediction can lead to bad actions.

The solution is to quantify uncertainty of the model, and take into consideration this uncertain in planning.

First of all, it's important to know that uncertainty of a model is not the same thing as the probability of the model's prediction on some state. Uncertainty is not about the setting where dynamics is noisy, but about the setting where we don't know what the dynamics are.

The way to avoid taking risky actions on uncertain state is to plan based on *expected expected reward*. Wait, what is it? Yes, this is not a typo, the first expected is with respect to the model uncertainty, and the second expected is with respect to trajectory distribution. Mathematically, the objective is

$$\begin{align}
&\int_{\theta} \int_{\tau} \sum_t r(s_t, a_t) p_{\theta}(\tau) p(\theta) \text{d}\tau  \text{d}\theta  \\
&= \int_{\theta} \left[\mathbb{E}_{\tau\sim p_{\theta}(\tau)}\sum_t r(s_t, a_t)\right] p(\theta)\text{d}\theta  \\
&= \mathbb{E}_{\theta\sim p(\theta)}\left[ \mathbb{E}_{\tau\sim p_{\theta}(\tau)}\sum_t r(s_t, a_t) \right]
\end{align}$$

Having an uncertainty-aware formulation, the next steps are: 

1. how do we get $$p(\theta)$$
2. how do we actually plan actions to optimize this objective
